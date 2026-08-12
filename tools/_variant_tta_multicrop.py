"""다중 스케일 TTA — 중앙 크롭 여러 배율을 합치면 '작게 찍힌 피사체'가 살아나는가(scratch).

이미 측정된 것: 피사체가 프레임을 꽉 채우면 5품목 top-1 100%,
차지비율이 50%→35%→25% 로 줄면 top-1 이 75%→55%→57% 로 무너진다.
가설: 작아진 것이 문제라면 '중앙을 잘라 키워서' 다시 물어보면 살아난다.

크롭을 어디서 자르느냐로 두 갈래가 갈린다. 둘은 성능도 비용도 다르므로 따로 잰다.

    server : 서버가 실제로 받는 224x224 에서 자른다 → 클라이언트 변경 0, 새 화소 정보 0(흐려짐)
    client : 브라우저가 원본 카메라 프레임에서 잘라 224 로 보낸다 → 화소는 살지만 클라 변경 필요

    python tools/_variant_tta_multicrop.py --self-test   # 측정기부터 검증(역검증 포함)
    python tools/_variant_tta_multicrop.py               # 본 측정
    python tools/_variant_tta_multicrop.py --n 30 --skip-probe

표본은 baseline 과 반드시 같다: E.sample_images(30) (eval_realworld 기본 seed=7).
scales=(1.0,) 로 돌리면 어떤 융합식을 써도 baseline 과 완전히 같은 숫자가 나와야 한다 — 자체검증에서 확인한다.

선택 기준은 결과를 보기 전에 못박는다(사후에 좋아 보이는 칸을 고르지 않기 위해):
    1순위  5개 조건(studio/far/fill50/fill35/fill25) 평균 top-3 최대
    제약   studio top-1 이 baseline 보다 2%p 넘게 떨어지면 탈락 (시연의 정상 경로를 깨면 안 된다)
    동점   평균 top-1, 그 다음 전방통과 횟수가 적은 쪽
"""

from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_realworld as E  # noqa: E402

import onnxruntime as ort  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ITEMS = E.ITEMS
SESSION = ort.InferenceSession(str(E.MODEL), providers=["CPUExecutionProvider"])
OUT_NAMES = [o.name for o in SESSION.get_outputs()]

GRAY = (150, 148, 143)          # eval_realworld.cond_far 가 쓰는 배경색
SCALES_ALL = (1.00, 0.70, 0.50, 0.35, 0.25)

# 이름 붙인 스케일 사다리 — 전방통과 횟수 = 길이
LADDERS: dict[str, tuple[float, ...]] = {
    "1.0(baseline)":          (1.00,),
    "1.0+0.5":                (1.00, 0.50),
    "1.0+0.35":               (1.00, 0.35),
    "1.0+0.25":               (1.00, 0.25),
    "1.0+0.5+0.35":           (1.00, 0.50, 0.35),
    "1.0+0.35+0.25":          (1.00, 0.35, 0.25),
    "1.0+0.7+0.5":            (1.00, 0.70, 0.50),
    "1.0+0.7+0.5+0.35":       (1.00, 0.70, 0.50, 0.35),
    "1.0+0.7+0.5+0.35+0.25":  (1.00, 0.70, 0.50, 0.35, 0.25),
}
MAIN_LADDER = "1.0+0.7+0.5+0.35"
STUDIO_TOLERANCE = 0.02         # studio top-1 허용 낙폭


# ── 표 출력 도우미(한글 폭 보정) ──────────────────────────────────────
def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, width: int, right: bool = False) -> str:
    fill = " " * max(0, width - _w(s))
    return fill + s if right else s + fill


# ── 전처리 ────────────────────────────────────────────────────────────
def center_crop(img: Image.Image, r: float) -> Image.Image:
    """중앙에서 변 길이 r 배만큼 잘라낸다. r=1.0 이면 원본 그대로."""
    if r >= 0.999:
        return img
    w, h = img.size
    cw, ch = max(1, round(w * r)), max(1, round(h * r))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def served_224(img: Image.Image) -> Image.Image:
    """서버가 실제로 받는 것 — 브라우저가 비율 무시하고 224x224 로 눌러 보낸 픽셀."""
    return img.convert("RGB").resize((E.SIZE, E.SIZE), Image.BILINEAR)


def tensor_at(img: Image.Image, r: float, mode: str) -> np.ndarray:
    """배율 r 의 입력 텐서. mode='server' 는 224 를 받은 뒤 자르고, 'client' 는 원본에서 자른다."""
    if r >= 0.999:
        return E.to_tensor(img)                      # 두 갈래 모두 baseline 과 같은 입력
    if mode == "server":
        return E.to_tensor(center_crop(served_224(img), r))
    return E.to_tensor(center_crop(img, r))


# ── 시연 조건 ─────────────────────────────────────────────────────────
def compose(img: Image.Image, occ: float, bg: tuple[int, int, int] = GRAY) -> Image.Image:
    """피사체가 프레임의 occ(선형 비율)만 차지하도록 배경 위에 올린다(_diag_potato.compose 와 동일)."""
    if occ >= 0.999:
        return img
    w, h = img.size
    canvas = Image.new("RGB", (round(w / occ), round(h / occ)), bg)
    canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


def compose_off(img: Image.Image, occ: float, fx: float = 0.28, fy: float = 0.28,
                bg: tuple[int, int, int] = GRAY) -> Image.Image:
    """같은 크기인데 피사체가 중앙이 아니라 (fx, fy) 위치에 있는 경우.

    중앙 크롭 TTA 의 전제('피사체는 가운데 있다')가 깨지는 지점을 재기 위한 대조군.
    """
    if occ >= 0.999:
        return img
    w, h = img.size
    W, H = round(w / occ), round(h / occ)
    canvas = Image.new("RGB", (W, H), bg)
    x = int(np.clip(fx * W - w / 2, 0, W - w))
    y = int(np.clip(fy * H - h / 2, 0, H - h))
    canvas.paste(img, (x, y))
    return canvas


CONDITIONS: dict[str, tuple[str, object]] = {
    "studio": ("학습과 같은 조건", lambda im: im),
    "far":    ("멀리서 촬영(far)", E.cond_far),
    "fill50": ("차지 50%", lambda im: compose(im, 0.50)),
    "fill35": ("차지 35%", lambda im: compose(im, 0.35)),
    "fill25": ("차지 25%", lambda im: compose(im, 0.25)),
}
PROBES: dict[str, tuple[str, object]] = {
    "off25":  ("차지 25% · 피사체가 좌상단으로 치우침", lambda im: compose_off(im, 0.25)),
    "off35":  ("차지 35% · 피사체가 좌상단으로 치우침", lambda im: compose_off(im, 0.35)),
}


# ── 모델 ──────────────────────────────────────────────────────────────
def item_logits(tensors: list[np.ndarray]) -> np.ndarray:
    x = np.concatenate(tensors, axis=0)
    out = SESSION.run(None, {SESSION.get_inputs()[0].name: x})
    return np.asarray(out[OUT_NAMES.index("item_logits")])


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


_CACHE: dict[tuple, np.ndarray] = {}


def logits_for(cond_key: str, item: str, files: list[Path], mode: str, r: float,
               table: dict[str, tuple[str, object]]) -> np.ndarray:
    """(조건, 품목, 갈래, 배율, 표본) 한 칸의 item_logits [n, C]. 같은 칸은 한 번만 돌린다.

    표본(파일 목록)까지 키에 넣는다 — 안 넣었더니 6장짜리 자체검증 결과가
    8장짜리 역검증으로 새어 들어와 엉뚱한 비교가 됐다(역검증이 잡아낸 실제 버그).
    """
    key = (cond_key, item, "-" if r >= 0.999 else mode, round(r, 3),
           hash(tuple(str(f) for f in files)))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    fn = table[cond_key][1]
    tens = [tensor_at(fn(Image.open(f).convert("RGB")), r, mode) for f in files]
    lg = item_logits(tens)
    _CACHE[key] = lg
    return lg


def stack_scales(cond_key: str, item: str, files: list[Path], mode: str,
                 scales: tuple[float, ...], table: dict[str, tuple[str, object]]) -> np.ndarray:
    """[n, S, C] — 배율마다의 logits 를 쌓는다."""
    return np.stack([logits_for(cond_key, item, files, mode, r, table) for r in scales], axis=1)


# ── 융합식 ────────────────────────────────────────────────────────────
def _weighted(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 1e-9)
    return (P * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def _entropy(P: np.ndarray) -> np.ndarray:
    return -(P * np.log(P + 1e-12)).sum(axis=-1)


def _margin(P: np.ndarray) -> np.ndarray:
    s = np.sort(P, axis=-1)
    return s[..., -1] - s[..., -2]


def f_mean(P, L):        return P.mean(axis=1)                                   # noqa: E704
def f_max(P, L):         return P.max(axis=1)                                    # noqa: E704
def f_logit_mean(P, L):  return softmax(L.mean(axis=1))                          # noqa: E704
def f_logit_max(P, L):   return softmax(L.max(axis=1))                           # noqa: E704
def f_conf(P, L):        return _weighted(P, P.max(axis=-1))                     # noqa: E704
def f_conf2(P, L):       return _weighted(P, P.max(axis=-1) ** 2)                # noqa: E704
def f_conf4(P, L):       return _weighted(P, P.max(axis=-1) ** 4)                # noqa: E704
def f_margin(P, L):      return _weighted(P, _margin(P))                         # noqa: E704
def f_entropy(P, L):     return _weighted(P, np.exp(-_entropy(P) / 0.5))         # noqa: E704


def f_best(P, L):
    """가장 자신 있는 배율 하나만 채택(winner-take-all)."""
    idx = P.max(axis=-1).argmax(axis=1)
    return P[np.arange(P.shape[0]), idx]


FUSIONS: dict[str, tuple[str, object]] = {
    "mean":       ("확률 평균", f_mean),
    "max":        ("확률 최대", f_max),
    "logit_mean": ("logit 평균(기하평균)", f_logit_mean),
    "logit_max":  ("logit 최대", f_logit_max),
    "conf_w":     ("신뢰도 가중(w=최대확률)", f_conf),
    "conf_w2":    ("신뢰도 가중 제곱", f_conf2),
    "conf_w4":    ("신뢰도 가중 4제곱", f_conf4),
    "margin_w":   ("여유 가중(1위-2위)", f_margin),
    "entropy_w":  ("엔트로피 가중", f_entropy),
    "best_conf":  ("최고신뢰 배율 단독", f_best),
}


# ── 지표 ──────────────────────────────────────────────────────────────
def topk(fused: np.ndarray, truth: str, k: int) -> float:
    order = np.argsort(-fused, axis=1)[:, :k]
    return float((order == ITEMS.index(truth)).any(axis=1).mean())


def evaluate(picks: dict[str, list[Path]], cond_key: str, mode: str, scales: tuple[float, ...],
             fusion: str, table: dict[str, tuple[str, object]] = CONDITIONS) -> dict[str, tuple[float, float]]:
    """조건 한 개에 대해 품목별 (top-1, top-3). 'overall' 은 품목 평균."""
    fn = FUSIONS[fusion][1]
    out: dict[str, tuple[float, float]] = {}
    for item, files in picks.items():
        L = stack_scales(cond_key, item, files, mode, scales, table)
        P = softmax(L)
        fused = fn(P, L)
        out[item] = (topk(fused, item, 1), topk(fused, item, 3))
    out["overall"] = (float(np.mean([v[0] for k, v in out.items() if k != "overall"])),
                      float(np.mean([v[1] for k, v in out.items() if k != "overall"])))
    return out


# ── 자체검증 ──────────────────────────────────────────────────────────
def _swatch() -> Image.Image:
    """회색 224 판 위 중앙에 변 길이 40% 인 빨간 사각형."""
    im = Image.new("RGB", (224, 224), (128, 128, 128))
    s = int(224 * 0.4)
    xy = (224 - s) // 2
    ImageDraw.Draw(im).rectangle([xy, xy, xy + s - 1, xy + s - 1], fill=(220, 20, 20))
    return im


def _red_frac(im: Image.Image) -> float:
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    return float(((a[..., 0] > 180) & (a[..., 1] < 80)).mean())


def _self_test(picks: dict[str, list[Path]]) -> None:
    checks = 0

    # 1) 크롭이 아는 답을 내는가 — 40% 사각형은 크롭 35% 안에서 화면을 가득 채워야 한다
    sw = _swatch()
    assert abs(_red_frac(sw) - 0.16) < 0.02, f"합성 판이 이상하다: {_red_frac(sw):.3f}"
    assert _red_frac(center_crop(sw, 0.35)) > 0.99, "0.35 크롭이 사각형 안으로 안 들어갔다"
    assert abs(_red_frac(center_crop(sw, 0.8)) - 0.25) < 0.03, "0.8 크롭 면적이 안 맞는다"
    assert center_crop(sw, 1.0).size == sw.size, "r=1.0 은 원본이어야 한다"
    checks += 3

    # 2) served_224 는 무엇이 들어와도 224x224 를 내놓는다(서버가 받는 것)
    assert served_224(Image.new("RGB", (896, 672))).size == (224, 224)
    checks += 1

    # 3) r=1.0 텐서는 baseline 전처리와 완전히 같아야 한다(두 갈래 모두)
    probe = Image.open(picks[ITEMS[0]][0]).convert("RGB")
    base_t = E.to_tensor(probe)
    for mode in ("server", "client"):
        assert np.array_equal(tensor_at(probe, 1.0, mode), base_t), f"{mode} r=1.0 이 baseline 과 다르다"
    checks += 2

    # 4) 융합식 아는 답 — 평균과 최고신뢰가 서로 다른 답을 내는 경우로 구분한다
    P = np.array([[[0.60, 0.20, 0.20], [0.30, 0.65, 0.05]]])
    L = np.log(P)
    assert f_mean(P, L).argmax(1)[0] == 0, "평균은 0번을 골라야 한다"
    assert f_max(P, L).argmax(1)[0] == 1, "최대는 1번을 골라야 한다"
    assert f_best(P, L).argmax(1)[0] == 1, "최고신뢰 배율은 1번을 골라야 한다"
    assert f_conf4(P, L).argmax(1)[0] == 1, "4제곱 가중은 1번 쪽으로 기울어야 한다"
    assert f_conf(P, L).argmax(1)[0] == 0, "1제곱 가중은 아직 0번이어야 한다"
    checks += 5

    # 5) 배율 순서를 바꿔도 결과가 같아야 한다(순서 의존 버그 잡기)
    small = {k: v[:6] for k, v in picks.items()}
    a = evaluate(small, "fill25", "server", (1.00, 0.50, 0.35), "mean")["overall"]
    b = evaluate(small, "fill25", "server", (0.35, 1.00, 0.50), "mean")["overall"]
    assert a == b, f"배율 순서에 따라 결과가 달라진다: {a} vs {b}"
    checks += 1

    # 6) scales=(1.0,) 이면 어떤 융합식이든 baseline(단일 argmax)과 같은 숫자여야 한다
    for item, files in small.items():
        lg = logits_for("far", item, files, "server", 1.0, CONDITIONS)
        want = (topk(lg, item, 1), topk(lg, item, 3))
        for name in FUSIONS:
            got = evaluate(small, "far", "server", (1.00,), name)[item]
            assert got == want, f"{name}: 단일 배율인데 baseline 과 다르다 {got} != {want}"
    checks += 1

    print(f"✅ 자체검증 {checks}건 통과 — 크롭 기하, 서버 입력, baseline 동치, 융합식 아는 답, 순서 무관")


def _fail_test(picks: dict[str, list[Path]]) -> None:
    """검사가 헛돌지 않는가 — 크롭을 고장 내면 TTA 이득이 반드시 0 이 되어야 한다."""
    global center_crop
    small = {k: v[:8] for k, v in picks.items()}
    scales = (1.00, 0.50, 0.35)

    real = evaluate(small, "fill25", "server", scales, "mean")["overall"]
    base = evaluate(small, "fill25", "server", (1.00,), "mean")["overall"]

    keep, cache_keep = center_crop, dict(_CACHE)
    try:
        center_crop = lambda img, r: img  # noqa: E731  크롭이 아무 일도 안 하게 만든다
        _CACHE.clear()
        broken = evaluate(small, "fill25", "server", scales, "mean")["overall"]
    finally:
        center_crop = keep
        _CACHE.clear()
        _CACHE.update(cache_keep)

    assert broken == base, f"크롭을 껐는데도 baseline 과 다르다 — 측정 경로가 샌다: {broken} vs {base}"
    assert real != broken, f"크롭이 살아 있는데 결과가 baseline 과 똑같다 — TTA 가 안 걸렸다: {real}"
    print(f"✅ 역검증 — 크롭을 끄면 {broken[0]:.0%}/{broken[1]:.0%}(=baseline), 켜면 {real[0]:.0%}/{real[1]:.0%}")


def _cross_check(picks: dict[str, list[Path]]) -> None:
    """이미 알려진 baseline 숫자를 같은 표본에서 재현하는지 대조한다(못 하면 표본이 다른 것)."""
    known_far = {"사과": 1.00, "양파": 0.83, "배": 0.60, "감귤": 0.53, "감자": 0.00}
    print("\n[baseline 재현 대조] 같은 표본·같은 전처리로 다시 잰 값")
    print(f"  {pad('조건', 10)}{pad('품목별 top-1', 40)}")
    for key in ("studio", "far"):
        row = evaluate(picks, key, "server", (1.00,), "mean")
        cells = "  ".join(f"{i} {row[i][0]:.0%}" for i in ITEMS)
        print(f"  {pad(CONDITIONS[key][0], 10)}{cells}")
        if key == "far":
            gap = max(abs(row[i][0] - known_far[i]) for i in ITEMS)
            mark = "일치" if gap <= 0.02 else f"어긋남 {gap:.0%}p"
            print(f"  {pad('', 10)}→ 기존 측정(사과100/양파83/배60/감귤53/감자0)과 {mark}")


# ── 표 ────────────────────────────────────────────────────────────────
def table_fusions(picks: dict[str, list[Path]], mode: str, ladder: str) -> dict[str, dict]:
    scales = LADDERS[ladder]
    print("\n" + "=" * 96)
    print(f"융합식 비교 · {mode} 갈래 · 배율 {ladder} (전방통과 {len(scales)}회/장)  — 각 칸 top-1/top-3")
    print("=" * 96)
    head = pad("융합식", 24) + "".join(pad(CONDITIONS[k][0], 14, right=True) for k in CONDITIONS) + pad("평균", 14, right=True)
    print(head)
    print("-" * _w(head))

    rows: dict[str, dict] = {}
    base = {k: evaluate(picks, k, mode, (1.00,), "mean") for k in CONDITIONS}
    b_cells = "".join(pad(f"{base[k]['overall'][0]:.0%}/{base[k]['overall'][1]:.0%}", 14, right=True) for k in CONDITIONS)
    b_avg = (float(np.mean([base[k]["overall"][0] for k in CONDITIONS])),
             float(np.mean([base[k]["overall"][1] for k in CONDITIONS])))
    print(pad("baseline(1회)", 24) + b_cells + pad(f"{b_avg[0]:.0%}/{b_avg[1]:.0%}", 14, right=True))

    for name, (label, _) in FUSIONS.items():
        per = {k: evaluate(picks, k, mode, scales, name) for k in CONDITIONS}
        avg = (float(np.mean([per[k]["overall"][0] for k in CONDITIONS])),
               float(np.mean([per[k]["overall"][1] for k in CONDITIONS])))
        rows[name] = {"per": per, "avg": avg}
        cells = "".join(pad(f"{per[k]['overall'][0]:.0%}/{per[k]['overall'][1]:.0%}", 14, right=True) for k in CONDITIONS)
        print(pad(label, 24) + cells + pad(f"{avg[0]:.0%}/{avg[1]:.0%}", 14, right=True))
    rows["_baseline"] = {"per": base, "avg": b_avg}
    return rows


def table_ladders(picks: dict[str, list[Path]], mode: str, fusion: str) -> None:
    print("\n" + "=" * 96)
    print(f"배율 사다리별 비용 대비 효과 · {mode} 갈래 · 융합식 '{FUSIONS[fusion][0]}'  — 각 칸 top-1/top-3")
    print("=" * 96)
    head = pad("배율", 26) + pad("통과", 6, right=True) + "".join(
        pad(CONDITIONS[k][0], 14, right=True) for k in CONDITIONS) + pad("평균", 14, right=True)
    print(head)
    print("-" * _w(head))
    for name, scales in LADDERS.items():
        f = "mean" if len(scales) == 1 else fusion
        per = {k: evaluate(picks, k, mode, scales, f) for k in CONDITIONS}
        avg = (float(np.mean([per[k]["overall"][0] for k in CONDITIONS])),
               float(np.mean([per[k]["overall"][1] for k in CONDITIONS])))
        cells = "".join(pad(f"{per[k]['overall'][0]:.0%}/{per[k]['overall'][1]:.0%}", 14, right=True) for k in CONDITIONS)
        print(pad(name, 26) + pad(str(len(scales)), 6, right=True) + cells
              + pad(f"{avg[0]:.0%}/{avg[1]:.0%}", 14, right=True))


def pick_best(rows_by_mode: dict[str, dict[str, dict]], picks: dict[str, list[Path]]) -> tuple[str, str, str]:
    """못박아 둔 기준으로 (갈래, 융합식, 사다리)를 고른다."""
    base_studio = rows_by_mode["server"]["_baseline"]["per"]["studio"]["overall"][0]
    best: tuple | None = None
    for mode in rows_by_mode:
        for ladder, scales in LADDERS.items():
            if len(scales) == 1:
                continue
            for name in FUSIONS:
                per = {k: evaluate(picks, k, mode, scales, name) for k in CONDITIONS}
                studio = per["studio"]["overall"][0]
                if studio < base_studio - STUDIO_TOLERANCE:
                    continue
                t3 = float(np.mean([per[k]["overall"][1] for k in CONDITIONS]))
                t1 = float(np.mean([per[k]["overall"][0] for k in CONDITIONS]))
                cand = (round(t3, 6), round(t1, 6), -len(scales))
                if best is None or cand > best[0]:
                    best = (cand, mode, name, ladder)
    if best is None:
        raise SystemExit("기준을 통과한 조합이 없다 — studio 를 지키면서 나아지는 구성이 없다")
    return best[1], best[2], best[3]


def table_per_item(picks: dict[str, list[Path]], mode: str, fusion: str, ladder: str) -> dict:
    scales = LADDERS[ladder]
    print("\n" + "=" * 96)
    print(f"채택 구성의 품목별 성적 · {mode} · {FUSIONS[fusion][0]} · 배율 {ladder} ({len(scales)}회)")
    print("=" * 96)
    head = pad("조건", 16) + "".join(pad(i, 16, right=True) for i in ITEMS) + pad("전체", 16, right=True)
    print(head)
    print("-" * _w(head))
    out: dict = {}
    for k, (label, _) in CONDITIONS.items():
        b = evaluate(picks, k, mode, (1.00,), "mean")
        t = evaluate(picks, k, mode, scales, fusion)
        out[k] = {"baseline": b, "tta": t}
        print(pad(label + " · 지금", 16) + "".join(
            pad(f"{b[i][0]:.0%}/{b[i][1]:.0%}", 16, right=True) for i in ITEMS)
            + pad(f"{b['overall'][0]:.0%}/{b['overall'][1]:.0%}", 16, right=True))
        print(pad(label + " · TTA", 16) + "".join(
            pad(f"{t[i][0]:.0%}/{t[i][1]:.0%}", 16, right=True) for i in ITEMS)
            + pad(f"{t['overall'][0]:.0%}/{t['overall'][1]:.0%}", 16, right=True))
    return out


def table_potato(picks: dict[str, list[Path]], mode: str, fusion: str, ladder: str) -> None:
    print("\n" + "=" * 96)
    print("감자(far) 단독 — 지금 top-1/top-3 이 0%/0% 인 지점")
    print("=" * 96)
    files = picks["감자"]
    print(pad("구성", 40) + pad("top-1", 10, right=True) + pad("top-3", 10, right=True)
          + pad("감자 평균순위", 16, right=True) + "   최빈 오답")
    for label, mo, sc, fu in [("지금(1회)", mode, (1.00,), "mean")] + [
            (f"TTA {name} ({len(s)}회)", mode, s, fusion) for name, s in LADDERS.items() if len(s) > 1]:
        L = stack_scales("far", "감자", files, mo, sc, CONDITIONS)
        P = softmax(L)
        fused = FUSIONS[fu][1](P, L)
        rank = float(np.mean([list(np.argsort(-f)).index(ITEMS.index("감자")) + 1 for f in fused]))
        pred = [ITEMS[i] for i in fused.argmax(1)]
        wrong = [p for p in pred if p != "감자"]
        top = max(set(wrong), key=wrong.count) if wrong else "-"
        print(pad(label, 40) + pad(f"{topk(fused, '감자', 1):.0%}", 10, right=True)
              + pad(f"{topk(fused, '감자', 3):.0%}", 10, right=True)
              + pad(f"{rank:.2f}위", 16, right=True) + f"   {top}")


def table_probe(picks: dict[str, list[Path]], mode: str, fusion: str, ladder: str) -> None:
    print("\n" + "=" * 96)
    print("리스크 대조군 — 피사체가 중앙에 없으면? (중앙 크롭 TTA 의 전제가 깨지는 지점)")
    print("=" * 96)
    scales = LADDERS[ladder]
    head = pad("조건", 40) + pad("지금 top-1/3", 18, right=True) + pad("TTA top-1/3", 18, right=True)
    print(head)
    print("-" * _w(head))
    for k, (label, _) in PROBES.items():
        b = evaluate(picks, k, mode, (1.00,), "mean", PROBES)
        t = evaluate(picks, k, mode, scales, fusion, PROBES)
        print(pad(label, 40) + pad(f"{b['overall'][0]:.0%}/{b['overall'][1]:.0%}", 18, right=True)
              + pad(f"{t['overall'][0]:.0%}/{t['overall'][1]:.0%}", 18, right=True))


def table_cost(picks: dict[str, list[Path]], mode: str, ladder: str) -> None:
    print("\n" + "=" * 96)
    print("연산량 실측 — CPU, 한 장을 배치 1회로 처리")
    print("=" * 96)
    imgs = [Image.open(f).convert("RGB") for f in picks[ITEMS[0]][:20]]
    far = [CONDITIONS["far"][1](im) for im in imgs]
    base_ms = 0.0
    for name, scales in [("1.0(baseline)", (1.00,)), (ladder, LADDERS[ladder]),
                         ("1.0+0.7+0.5+0.35+0.25", SCALES_ALL)]:
        for _ in range(2):                     # 예열
            item_logits([tensor_at(far[0], r, mode) for r in scales])
        t0 = time.perf_counter()
        for im in far:
            item_logits([tensor_at(im, r, mode) for r in scales])
        dt = (time.perf_counter() - t0) / len(far) * 1000
        if not base_ms:
            base_ms = dt
            tail = "(기준)"
        else:
            tail = f"(baseline 대비 {dt / base_ms:.2f}배)"
        print(f"  {pad(name, 26)}전방통과 {len(scales)}회/장   {dt:6.1f} ms/장  {tail}")


FOCUS = ["1.0(baseline)", "1.0+0.35", "1.0+0.25", "1.0+0.35+0.25"]


def run_focus(picks: dict[str, list[Path]]) -> None:
    """본 측정에서 '가운데 배율(0.7·0.5)은 일을 안 한다'가 드러난 뒤 좁혀서 다시 재는 판.

    싼 사다리만 남기고, 실제로 배치할 후보의 연산량까지 같이 잰다.
    """
    for mode in ("server", "client"):
        for fusion in ("max", "mean"):
            print("\n" + "=" * 96)
            print(f"좁힌 비교 · {mode} · 융합식 '{FUSIONS[fusion][0]}'  — 각 칸 top-1/top-3")
            print("=" * 96)
            head = pad("배율", 20) + pad("통과", 6, right=True) + "".join(
                pad(CONDITIONS[k][0], 14, right=True) for k in CONDITIONS) + pad("평균", 14, right=True)
            print(head)
            print("-" * _w(head))
            for name in FOCUS:
                scales = LADDERS[name]
                f = "mean" if len(scales) == 1 else fusion
                per = {k: evaluate(picks, k, mode, scales, f) for k in CONDITIONS}
                avg = (float(np.mean([per[k]["overall"][0] for k in CONDITIONS])),
                       float(np.mean([per[k]["overall"][1] for k in CONDITIONS])))
                cells = "".join(pad(f"{per[k]['overall'][0]:.0%}/{per[k]['overall'][1]:.0%}", 14, right=True)
                                for k in CONDITIONS)
                print(pad(name, 20) + pad(str(len(scales)), 6, right=True) + cells
                      + pad(f"{avg[0]:.0%}/{avg[1]:.0%}", 14, right=True))

    for mode in ("server", "client"):
        print("\n" + "=" * 96)
        print(f"감자(far) · {mode} · 확률 최대")
        print("=" * 96)
        files = picks["감자"]
        print(pad("구성", 26) + pad("top-1", 10, right=True) + pad("top-3", 10, right=True)
              + pad("감자 평균순위", 16, right=True))
        for name in FOCUS:
            scales = LADDERS[name]
            f = "mean" if len(scales) == 1 else "max"
            L = stack_scales("far", "감자", files, mode, scales, CONDITIONS)
            P = softmax(L)
            fused = FUSIONS[f][1](P, L)
            rank = float(np.mean([list(np.argsort(-x)).index(ITEMS.index("감자")) + 1 for x in fused]))
            print(pad(name, 26) + pad(f"{topk(fused, '감자', 1):.0%}", 10, right=True)
                  + pad(f"{topk(fused, '감자', 3):.0%}", 10, right=True) + pad(f"{rank:.2f}위", 16, right=True))

    for mode in ("server", "client"):
        print("\n" + "=" * 96)
        print(f"리스크 대조군(피사체가 중앙에 없음) · {mode} · 확률 최대 · 1.0+0.35+0.25")
        print("=" * 96)
        for k, (label, _) in PROBES.items():
            b = evaluate(picks, k, mode, (1.00,), "mean", PROBES)
            t = evaluate(picks, k, mode, LADDERS["1.0+0.35+0.25"], "max", PROBES)
            print(pad(label, 40) + pad(f"지금 {b['overall'][0]:.0%}/{b['overall'][1]:.0%}", 18, right=True)
                  + pad(f"TTA {t['overall'][0]:.0%}/{t['overall'][1]:.0%}", 18, right=True))

    print("\n" + "=" * 96)
    print("연산량 실측 — CPU, 한 장을 배치 1회로 처리(배포 후보 위주)")
    print("=" * 96)
    imgs = [Image.open(f).convert("RGB") for f in picks[ITEMS[0]][:20]]
    far = [CONDITIONS["far"][1](im) for im in imgs]
    base_ms = 0.0
    for name in ["1.0(baseline)", "1.0+0.35", "1.0+0.35+0.25", "1.0+0.7+0.5+0.35+0.25"]:
        scales = LADDERS[name]
        for _ in range(2):
            item_logits([tensor_at(far[0], r, "client") for r in scales])
        t0 = time.perf_counter()
        for im in far:
            item_logits([tensor_at(im, r, "client") for r in scales])
        dt = (time.perf_counter() - t0) / len(far) * 1000
        if not base_ms:
            base_ms, tail = dt, "(기준)"
        else:
            tail = f"({dt / base_ms:.2f}배)"
        print(f"  {pad(name, 26)}전방통과 {len(scales)}회/장   {dt:6.1f} ms/장  {tail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="품목당 표본 수(baseline 과 같아야 한다)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--focus", action="store_true", help="싼 사다리만 좁혀서 재측정")
    args = ap.parse_args()

    picks = E.sample_images(args.n)             # seed=7 고정 — baseline 과 같은 표본
    if not picks:
        raise SystemExit(f"검증 이미지가 없습니다: {E.VALID}")

    _self_test(picks)
    _fail_test(picks)
    if args.self_test:
        return 0

    n_total = sum(len(v) for v in picks.values())
    print(f"\n표본: 품목당 {args.n}장 × {len(picks)}품목 = {n_total}장 (E.sample_images(seed=7), baseline 과 동일)")
    _cross_check(picks)

    if args.focus:
        run_focus(picks)
        return 0

    rows = {mode: table_fusions(picks, mode, MAIN_LADDER) for mode in ("server", "client")}

    mode, fusion, ladder = pick_best(rows, picks)
    print("\n" + "#" * 96)
    print(f"# 못박아 둔 기준(평균 top-3 최대, studio top-1 낙폭 ≤{STUDIO_TOLERANCE:.0%})으로 고른 구성: "
          f"{mode} · {FUSIONS[fusion][0]} · 배율 {ladder}")
    print("#" * 96)

    for m in ("server", "client"):
        table_ladders(picks, m, fusion)
    table_per_item(picks, mode, fusion, ladder)
    table_potato(picks, mode, fusion, ladder)
    if mode != "server":                        # 클라 변경 없이 되는 갈래도 같이 남긴다
        table_per_item(picks, "server", fusion, ladder)
        table_potato(picks, "server", fusion, ladder)
    if not args.skip_probe:
        table_probe(picks, mode, fusion, ladder)
        if mode != "server":
            table_probe(picks, "server", fusion, ladder)
    table_cost(picks, mode, ladder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
