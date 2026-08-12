"""종횡비를 지키는 전처리 3종 — 정말 '멀리서 촬영'을 살리는가(scratch).

지금 브라우저는 원본을 비율 무시하고 224x224 로 늘린다(features.js drawImage).
그 대신 셋을 비교한다.

    (a) center_square   중앙 정사각 크롭 → 224 리사이즈
    (b) shortside_crop  짧은 변을 224 로 맞춰 리사이즈 → 중앙 224 크롭
    (c) letterbox       긴 변을 224 로 맞춰 리사이즈 → 남는 곳을 여백으로 채움

측정은 baseline 과 같은 표본(E.sample_images(30), seed=7)·같은 조건 합성으로 한다.
조건 합성은 _diag_potato.compose 와 픽셀 단위로 같은지 self-test 에서 대조한다.

    python tools/_variant_center_square.py --self-test   # 측정기부터 검증
    python tools/_variant_center_square.py               # 본 측정
    python tools/_variant_center_square.py --n 30 --json out.json

주의(이 측정의 핵심 전제):
    valid 이미지는 224x224 정사각이다. 정사각 입력에서는 (a)(b)(c) 가 모두
    '아무것도 자르지 않고 아무것도 덧대지 않는' 항등 연산이 되어 baseline 과 같아진다.
    far/차지비율 조건도 정사각 캔버스 위에 합성되므로 마찬가지다.
    그래서 이 파일은 정사각 조건(필수 보고분)과 함께,
    셋이 실제로 갈라지는 비정사각(폰 3:4) 조건을 따로 나눠 잰다.
"""

from __future__ import annotations

import argparse
import json
import sys
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
SIZE = E.SIZE
GRAY = (150, 148, 143)          # _diag_potato 의 far 배경색과 같다
MEAN_PAD = tuple(int(round(v * 255)) for v in E.MEAN)   # 정규화 후 0 이 되는 색

_SESSION: ort.InferenceSession | None = None


def session() -> ort.InferenceSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = ort.InferenceSession(str(E.MODEL), providers=["CPUExecutionProvider"])
    return _SESSION


# ── 전처리 변형 ────────────────────────────────────────────────────────
# 모두 '기하 변형 → E.to_tensor' 순서다. 마지막 단계(224 리사이즈+정규화)를
# baseline 과 똑같이 공유해야 변형 이외의 차이가 섞이지 않는다.

def v_stretch(img: Image.Image) -> np.ndarray:
    """baseline — 비율 무시하고 224x224 로 늘린다(현재 브라우저 동작)."""
    return E.to_tensor(img)


def crop_center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    x, y = (w - s) // 2, (h - s) // 2
    return img.crop((x, y, x + s, y + s))


def v_center_square(img: Image.Image) -> np.ndarray:
    """(a) 중앙 정사각 크롭 후 리사이즈."""
    return E.to_tensor(crop_center_square(img))


def v_shortside_crop(img: Image.Image) -> np.ndarray:
    """(b) 짧은 변 기준 리사이즈 후 중앙 크롭.

    (a) 와 기하학적으로 같은 영역을 보지만 리샘플 순서가 다르다.
    긴 변을 먼저 줄인 뒤 자르므로 (a) 보다 보간이 한 번 더 섞인다.
    """
    w, h = img.size
    k = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * k)), max(SIZE, round(h * k))
    r = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
    x, y = (nw - SIZE) // 2, (nh - SIZE) // 2
    return E.to_tensor(r.crop((x, y, x + SIZE, y + SIZE)))


def _edge_color(img: Image.Image) -> tuple[int, int, int]:
    a = np.asarray(img.convert("RGB"))
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0)
    return tuple(int(v) for v in np.median(border, axis=0))


def v_letterbox(img: Image.Image, pad: tuple[int, int, int] | str = MEAN_PAD) -> np.ndarray:
    """(c) 레터박스 — 비율을 지켜 통째로 넣고 남는 곳을 채운다. 피사체는 오히려 작아진다."""
    w, h = img.size
    s = SIZE / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
    fill = _edge_color(img) if pad == "edge" else pad
    canvas = Image.new("RGB", (SIZE, SIZE), fill)
    canvas.paste(r, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return E.to_tensor(canvas)


VARIANTS: dict[str, tuple[str, object]] = {
    "stretch":      ("baseline 늘리기", v_stretch),
    "center_sq":    ("(a) 중앙정사각크롭", v_center_square),
    "shortside":    ("(b) 짧은변+중앙크롭", v_shortside_crop),
    "letterbox":    ("(c) 레터박스(평균색)", v_letterbox),
    "letterbox_ed": ("(c') 레터박스(가장자리색)", lambda im: v_letterbox(im, "edge")),
}


# ── 조건 합성 ─────────────────────────────────────────────────────────
def compose(img: Image.Image, occ: float, bg: tuple[int, int, int] = GRAY) -> Image.Image:
    """_diag_potato.compose 와 같은 일 — 정사각 캔버스에서 피사체가 선형 occ 만 차지."""
    if occ >= 0.999:
        return img
    w, h = img.size
    canvas = Image.new("RGB", (round(w / occ), round(h / occ)), bg)
    canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


def compose_ar(img: Image.Image, occ: float, ar: float = 4 / 3,
               bg: tuple[int, int, int] = GRAY) -> Image.Image:
    """폰 세로(3:4) 프레임. 피사체는 '짧은 변(가로)'의 occ 만 차지한다.

    정사각 조건과 짧은 변 기준 크기를 똑같이 두었으므로,
    (a)(b) 가 얻는 이득은 순전히 '긴 변 여백을 버려서 생긴 확대'다.
    """
    w, h = img.size
    fw = max(w, round(w / occ))
    fh = round(fw * ar)
    canvas = Image.new("RGB", (fw, fh), bg)
    canvas.paste(img, ((fw - w) // 2, (fh - h) // 2))
    return canvas


CONDITIONS: dict[str, tuple[str, object]] = {
    # 필수 보고 조건 — 전부 정사각 입력
    "studio":  ("studio(원본)",        lambda im: im),
    "far":     ("far(멀리서)",          E.cond_far),
    "fill50":  ("차지 50%",            lambda im: compose(im, 0.50)),
    "fill35":  ("차지 35%",            lambda im: compose(im, 0.35)),
    "fill25":  ("차지 25%",            lambda im: compose(im, 0.25)),
    # 추가 조건 — 폰 세로 3:4(비정사각). 셋이 실제로 갈라지는 유일한 지점
    "p34_100": ("[3:4] 차지 100%",     lambda im: compose_ar(im, 1.00)),
    "p34_50":  ("[3:4] 차지 50%",      lambda im: compose_ar(im, 0.50)),
    "p34_35":  ("[3:4] 차지 35%",      lambda im: compose_ar(im, 0.35)),
    "p34_25":  ("[3:4] 차지 25%",      lambda im: compose_ar(im, 0.25)),
}
SQUARE_KEYS = ["studio", "far", "fill50", "fill35", "fill25"]
AR_KEYS = ["p34_100", "p34_50", "p34_35", "p34_25"]


# ── 추론 ──────────────────────────────────────────────────────────────
def logits(tensors: list[np.ndarray]) -> np.ndarray:
    s = session()
    x = np.concatenate(tensors, axis=0)
    out = s.run(None, {s.get_inputs()[0].name: x})
    names = [o.name for o in s.get_outputs()]
    return np.asarray(out[names.index("item_logits")])


def topk_rate(lg: np.ndarray, truth: str, k: int) -> float:
    order = np.argsort(-lg, axis=1)[:, :k]
    return float((order == ITEMS.index(truth)).any(axis=1).mean())


# ── 측정기 검증 ───────────────────────────────────────────────────────
def _bbox_of(img: Image.Image, bg: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    d = np.sqrt(((a - np.array(bg, dtype=np.float32)) ** 2).sum(axis=2))
    m = d > 30
    if m.sum() < 16:
        return None
    ys, xs = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _geom(fn, img: Image.Image) -> Image.Image:
    """변형의 '기하 부분'만 눈으로 볼 수 있게 224x224 이미지로 되돌린다."""
    t = fn(img)[0].transpose(1, 2, 0)
    a = np.clip((t * E.STD + E.MEAN) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def _self_test() -> None:
    checks = 0

    # 0) 조건 합성이 baseline 진단도구와 같은가 — 다른 조건을 재면 비교가 무의미하다
    import _diag_potato as P  # noqa: PLC0415
    probe = Image.open(next(iter(E.sample_images(1).values()))[0]).convert("RGB")
    for occ in (0.50, 0.35, 0.25):
        mine = np.asarray(compose(probe, occ))
        theirs = np.asarray(P.compose(probe, GRAY, occ))
        assert mine.shape == theirs.shape and (mine == theirs).all(), \
            f"occ={occ} 합성이 _diag_potato.compose 와 다르다"
        checks += 1

    # 1) 224 정사각을 224 로 리사이즈하는 것이 항등인가
    #    (b) 는 이미 224 로 자른 뒤 E.to_tensor 에 넘기므로 이게 깨지면 (b)가 오염된다
    assert np.array_equal(np.asarray(probe.resize((SIZE, SIZE), Image.BILINEAR)), np.asarray(probe)), \
        "224→224 리사이즈가 픽셀을 바꾼다"
    checks += 1

    # 2) 정사각 입력에서 네 변형이 모두 같은 텐서를 내는가 (= 이 측정의 결론)
    base = v_stretch(probe)
    for key, (label, fn) in VARIANTS.items():
        assert np.array_equal(fn(probe), base), f"정사각인데 {label} 가 baseline 과 다르다"
    checks += 1

    # 3) 비정사각 입력에서는 반드시 갈라져야 한다 — 안 갈라지면 변형이 헛돈 것이다
    tall = compose_ar(probe, 0.50)
    seen = {k: fn(tall) for k, (_, fn) in VARIANTS.items()}
    for k in ("center_sq", "shortside", "letterbox"):
        assert not np.array_equal(seen[k], seen["stretch"]), \
            f"3:4 입력인데 {k} 가 baseline 과 같다 — 변형이 동작하지 않는다"
    assert not np.array_equal(seen["center_sq"], seen["letterbox"]), "(a)와 (c)가 같다"
    checks += 1

    # 4) (a) 는 3:4 프레임에서 피사체를 정확히 4/3 배로 키워야 한다
    tall_probe = compose_ar(probe, 0.50)
    b_stretch = _bbox_of(_geom(v_stretch, tall_probe), GRAY)
    b_crop = _bbox_of(_geom(v_center_square, tall_probe), GRAY)
    assert b_stretch and b_crop
    w_s, w_c = b_stretch[2] - b_stretch[0], b_crop[2] - b_crop[0]
    assert abs(w_c / w_s - 1.0) < 0.06, f"가로폭이 변하면 안 된다: {w_s}→{w_c}"
    h_s, h_c = b_stretch[3] - b_stretch[1], b_crop[3] - b_crop[1]
    assert abs((h_c / h_s) / (4 / 3) - 1.0) < 0.08, \
        f"(a)가 세로를 4/3 배로 키우지 않는다: {h_s}→{h_c} (배율 {h_c/h_s:.2f})"
    checks += 1

    # 5) (c) 는 비율을 지켜야 한다 — 정사각 피사체가 정사각으로 남는가
    syn = Image.new("RGB", (300, 400), GRAY)
    ImageDraw.Draw(syn).rectangle([100, 150, 199, 249], fill=(210, 60, 40))   # 100x100 정사각
    bl = _bbox_of(_geom(lambda im: v_letterbox(im), syn), GRAY)
    assert bl and abs((bl[2] - bl[0]) / (bl[3] - bl[1]) - 1.0) < 0.10, \
        f"레터박스가 비율을 깼다: {bl}"
    checks += 1

    # 6) 음성 대조군 — baseline 은 반드시 비율을 깨야 한다(안 깨지면 5번이 헛통과다)
    bs = _bbox_of(_geom(v_stretch, syn), GRAY)
    assert bs and abs((bs[2] - bs[0]) / (bs[3] - bs[1]) - 1.0) > 0.15, \
        f"baseline 이 비율을 안 깬다 — 검사가 헛돈다: {bs}"
    checks += 1

    print(f"✅ 검증 {checks}건 — 조건 합성 동일성, 정사각 항등, 3:4 분기, (a)확대 4/3, (c)비율보존, baseline 왜곡")


def _fail_test() -> None:
    """일부러 망가뜨린 변형이 검사에 걸리는가."""
    probe = Image.open(next(iter(E.sample_images(1).values()))[0]).convert("RGB")
    tall = compose_ar(probe, 0.50)
    broken = lambda im: E.to_tensor(im)  # noqa: E731  크롭을 빼먹은 가짜 center_square
    assert np.array_equal(broken(tall), v_stretch(tall))
    try:
        assert not np.array_equal(broken(tall), v_stretch(tall)), "x"
    except AssertionError:
        print("✅ 역검증 — 크롭을 빼먹으면 3:4 분기 검사가 즉시 걸린다(검사가 살아 있다)")
        return
    raise AssertionError("역검증 실패 — 망가뜨린 변형이 통과했다")


# ── 본 측정 ───────────────────────────────────────────────────────────
def measure(n: int) -> dict:
    picks = E.sample_images(n)                       # baseline 과 같은 표본(seed=7)
    cache = {it: [Image.open(f).convert("RGB") for f in fs] for it, fs in picks.items()}
    res: dict = {}
    for ckey, (clabel, cfn) in CONDITIONS.items():
        staged = {it: [cfn(im) for im in ims] for it, ims in cache.items()}
        res[ckey] = {"label": clabel, "variants": {}}
        for vkey, (vlabel, vfn) in VARIANTS.items():
            per: dict[str, dict[str, float]] = {}
            for it, ims in staged.items():
                lg = logits([vfn(im) for im in ims])
                per[it] = {"top1": topk_rate(lg, it, 1), "top3": topk_rate(lg, it, 3)}
            res[ckey]["variants"][vkey] = {
                "label": vlabel,
                "per_item": per,
                "top1": float(np.mean([v["top1"] for v in per.values()])),
                "top3": float(np.mean([v["top3"] for v in per.values()])),
            }
    return res


def table(res: dict, keys: list[str], title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    w = max(len(v[0]) for v in VARIANTS.values()) + 2
    for ckey in keys:
        r = res[ckey]
        print(f"\n■ {r['label']}   각 칸 = top-1 / top-3")
        head = f"{'전처리':<{w}}{'전체':>13}   " + "".join(f"{i:>11}" for i in ITEMS)
        print(head)
        print("-" * len(head))
        for vkey, v in r["variants"].items():
            cells = "".join(f"{v['per_item'][i]['top1']:>5.0%}/{v['per_item'][i]['top3']:<5.0%}"
                            for i in ITEMS)
            mark = "  ←기준" if vkey == "stretch" else ""
            print(f"{v['label']:<{w}}{v['top1']:>6.0%}/{v['top3']:<6.0%}   {cells}{mark}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="품목당 표본 수(baseline 과 같게 30)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        _fail_test()
        return 0

    _self_test()
    _fail_test()

    res = measure(args.n)

    print(f"\n검증 분할 · 품목당 {args.n}장 · E.sample_images(seed=7) — baseline 과 같은 표본")
    table(res, SQUARE_KEYS, "필수 보고 조건 (입력이 정사각 224x224)")
    table(res, AR_KEYS, "추가 조건 — 폰 세로 3:4 프레임 (셋이 실제로 갈라지는 유일한 지점)")

    # 감자 far top-3 별도 보고
    print("\n" + "=" * 96)
    print("감자가 top-3 안에 들어오는가 (지금 0% 인 지점)")
    print("=" * 96)
    head = f"{'조건':<18}" + "".join(f"{VARIANTS[k][0]:>26}" for k in VARIANTS)
    print(head)
    print("-" * len(head))
    for ckey in SQUARE_KEYS + AR_KEYS:
        cells = "".join(
            f"{res[ckey]['variants'][k]['per_item']['감자']['top1']:>11.0%}"
            f" /{res[ckey]['variants'][k]['per_item']['감자']['top3']:>11.0%}"
            for k in VARIANTS)
        print(f"{res[ckey]['label']:<18}{cells}")

    # 결론 — baseline 대비 변화량
    print("\n" + "=" * 96)
    print("baseline 대비 전체 top-1 / top-3 변화 (%p)")
    print("=" * 96)
    head = f"{'조건':<18}" + "".join(f"{VARIANTS[k][0]:>26}" for k in VARIANTS if k != "stretch")
    print(head)
    print("-" * len(head))
    for ckey in SQUARE_KEYS + AR_KEYS:
        b = res[ckey]["variants"]["stretch"]
        cells = ""
        for k in VARIANTS:
            if k == "stretch":
                continue
            v = res[ckey]["variants"][k]
            cells += f"{100*(v['top1']-b['top1']):>+11.1f}/{100*(v['top3']-b['top3']):>+11.1f}"
        print(f"{res[ckey]['label']:<18}{cells}")

    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
