"""
시연장에서 실제로 일어날 법한, 아직 안 잰 촬영 조건들을 잰다.

이미 잰 것(비율 늘림·흙배경·어두움·기울임 = 전부 100%, 멀어지면 급락)은 다시 재지 않는다.
여기서 새로 재는 것은 '심사위원이 폰을 드는 순간 실제로 벌어지는 일'들이다:
여러 개를 한 번에, 상자째, 손으로 들고, 그림자 지고, 형광등 아래, 흔들리고,
초점 안 맞고, 카톡으로 보내 압축되고, 프레임 밖으로 잘려나가는 상황.

화면이 후보 3개를 보여주므로 top-1 과 top-3 를 같이 잰다.
(품목이 5종이라 아무렇게나 찍어도 top-3 는 60% 다. 그 아래면 무작위보다 못한 것이다.)

    python tools/_diag_conditions.py --self-test
    python tools/_diag_conditions.py --n 30
    python tools/_diag_conditions.py --n 30 --json out.json

원본 검증 이미지가 이미 224x224 라, 합성 조건은 피사체를 원본 크기 그대로 큰 캔버스에
올린 뒤 to_tensor 가 한 번만 축소하게 만든다(불필요한 재샘플링으로 손해 보지 않게).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_realworld as E  # noqa: E402  전처리·표본추출·모델실행을 그대로 재사용

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

RNG = np.random.default_rng(20260812)


# ── 도우미 ────────────────────────────────────────────────────────────
def _surface(w: int, h: int, rgb: tuple[int, int, int], noise: int = 10) -> Image.Image:
    """단색이 아닌 현실적인 바닥면 — 완전 단색이면 모델이 배경을 무시해버려 조건이 헐거워진다."""
    base = np.full((h, w, 3), rgb, dtype=np.int16)
    base += RNG.integers(-noise, noise + 1, (h, w, 3), dtype=np.int16)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def cutout(img: Image.Image) -> Image.Image:
    """스튜디오 흰 배경을 떼어내 RGBA 로 돌려준다.

    합성 조건(여러 개·상자·손)에서 원본을 그대로 붙이면 과일 둘레에 흰 네모가 남는다.
    실제 사진에는 없는 무늬이고, 흰 배경은 학습 조건 그 자체라 모델을 부당하게 도와줄 수 있다.
    사과 표면의 흰 반사광은 남겨야 하므로, 테두리에서 이어진 흰 영역만 배경으로 본다.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    gray = a.mean(axis=2)
    flat = a.max(axis=2) - a.min(axis=2)          # 채도가 낮아야 배경(흰 종이)
    whitish = (gray >= 224) & (flat <= 30)

    lbl, n = ndimage_label(whitish)
    if n == 0:
        return img.convert("RGBA")
    border = np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]])
    bg_ids = set(int(v) for v in np.unique(border) if v != 0)
    bg = np.isin(lbl, list(bg_ids)) if bg_ids else np.zeros_like(whitish)

    share = bg.mean()
    if not (0.03 <= share <= 0.88):               # 떼어내기 실패 — 원본 그대로 쓴다
        return img.convert("RGBA")

    alpha = np.where(bg, 0, 255).astype(np.uint8)
    out = img.convert("RGBA")
    # 경계를 1픽셀 부드럽게 — 계단 무늬가 새 신호가 되지 않게
    out.putalpha(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(0.8)))
    return out


def ndimage_label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import ndimage
    return ndimage.label(mask)


def _paste(canvas: Image.Image, piece: Image.Image, xy: tuple[int, int]) -> None:
    canvas.paste(piece, xy, piece if piece.mode == "RGBA" else None)


def _tint(img: Image.Image, gains: tuple[float, float, float], lift: int = 0) -> Image.Image:
    """채널별 이득 — 색온도 흉내."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    a = a * np.array(gains, dtype=np.float32) + lift
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ── 조건 ──────────────────────────────────────────────────────────────
def cond_multi3(img: Image.Image) -> Image.Image:
    """사과 3개를 나란히 놓고 한 장에 담는다. 시연에서 가장 자연스러운 행동."""
    w, h = img.size
    piece = cutout(img)
    gap = w // 12
    canvas = _surface(w * 3 + gap * 4, int((w * 3 + gap * 4) * 3 / 4), (206, 200, 190))
    y = (canvas.height - h) // 2
    for i in range(3):
        _paste(canvas, piece, (gap + i * (w + gap), y))
    return canvas


def cond_multi1_ctrl(img: Image.Image) -> Image.Image:
    """대조군 — multi3 와 화면 차지 비율은 같지만 한 개만 놓는다.
    'multi3 가 나빠진 이유'가 여러 개라서인지, 그냥 작아서인지 가른다."""
    w, h = img.size
    gap = w // 12
    canvas = _surface(w * 3 + gap * 4, int((w * 3 + gap * 4) * 3 / 4), (206, 200, 190))
    _paste(canvas, cutout(img), ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


def cond_box(img: Image.Image) -> Image.Image:
    """상자에 담긴 채로 촬영. 골판지 테두리 + 3x3 로 채워진 상태."""
    w, h = img.size
    pad = w // 5
    inner_w, inner_h = w * 3, h * 3
    canvas = _surface(inner_w + pad * 2, inner_h + pad * 2, (150, 112, 74), noise=14)
    d = ImageDraw.Draw(canvas)
    d.rectangle([pad // 2, pad // 2, canvas.width - pad // 2, canvas.height - pad // 2],
                outline=(112, 80, 50), width=max(3, pad // 4))
    piece = cutout(img)
    for r in range(3):
        for c in range(3):
            _paste(canvas, piece, (pad + c * w, pad + r * h))
    return canvas


def cond_hand(img: Image.Image) -> Image.Image:
    """손으로 들고 찍기 — 손가락이 피사체 아래쪽을 가린다."""
    w, h = img.size
    canvas = _surface(int(w * 1.4), int(h * 1.4), (198, 194, 186))
    ox, oy = (canvas.width - w) // 2, (canvas.height - h) // 2
    _paste(canvas, cutout(img), (ox, oy))
    d = ImageDraw.Draw(canvas)
    skin, shade = (222, 176, 142), (196, 148, 116)
    fw = int(w * 0.20)
    top = oy + int(h * 0.62)
    for i in range(4):
        x0 = ox + int(w * 0.06) + i * int(w * 0.23)
        d.rounded_rectangle([x0, top + int(h * 0.05 * abs(i - 1.5)), x0 + fw, canvas.height],
                            radius=fw // 2, fill=skin, outline=shade, width=max(2, fw // 12))
    # 손바닥
    d.rounded_rectangle([ox - int(w * 0.05), oy + int(h * 0.92),
                         ox + int(w * 1.05), canvas.height],
                        radius=fw // 2, fill=skin)
    return canvas


def cond_shadow(img: Image.Image) -> Image.Image:
    """찍는 사람/폰 그림자가 피사체 위로 짙게 드리운 경우. 경계가 뚜렷한 반쪽 그림자."""
    w, h = img.size
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    xs = np.linspace(0, 1, w, dtype=np.float32)
    # 왼쪽 55% 는 0.32 배로 어둡고, 좁은 구간에서 급하게 밝아진다
    mask = 0.32 + 0.68 / (1 + np.exp(-(xs - 0.55) * 45))
    a *= mask[None, :, None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def cond_fluorescent(img: Image.Image) -> Image.Image:
    """창고·마트 형광등. 파랑-초록끼가 돌고 대비가 조금 눌린다."""
    out = _tint(img, (0.86, 1.02, 1.18), lift=6)
    a = np.asarray(out, dtype=np.float32)
    a = (a - 128) * 0.92 + 128  # 대비 살짝 감소
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def cond_incandescent(img: Image.Image) -> Image.Image:
    """백열등·해질녘. 노랑-주황끼. 감귤/양파 색과 겹쳐 위험할 수 있는 조건."""
    return _tint(img, (1.18, 0.98, 0.72), lift=4)


def cond_motionblur(img: Image.Image) -> Image.Image:
    """셔터 누르는 순간 흔들림 — 한 방향으로 번진다(가로 약 7% 길이)."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    k = max(3, int(img.size[0] * 0.07))
    acc = np.zeros_like(a)
    for s in range(k):
        acc += np.roll(a, s - k // 2, axis=1)
    return Image.fromarray(np.clip(acc / k, 0, 255).astype(np.uint8))


def cond_defocus(img: Image.Image) -> Image.Image:
    """가까이 대서 초점이 안 맞은 경우 — 등방성 흐림."""
    return img.filter(ImageFilter.GaussianBlur(radius=max(1.5, img.size[0] * 0.018)))


def cond_jpeg(img: Image.Image) -> Image.Image:
    """메신저로 보내며 강하게 압축된 사진(품질 12, 4:2:0)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=12, subsampling=2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def cond_crop(img: Image.Image) -> Image.Image:
    """피사체가 프레임 밖으로 잘려 오른쪽 아래 40% 가 안 보인다."""
    w, h = img.size
    return img.crop((0, 0, int(w * 0.62), int(h * 0.62)))


CONDITIONS: dict[str, tuple[str, object]] = {
    "studio": ("학습과 같은 조건(기준)", E.cond_studio),
    "multi3": ("3개 나란히", cond_multi3),
    "multi1_ctrl": ("1개·같은 크기(대조)", cond_multi1_ctrl),
    "box": ("상자에 담긴 채(3x3)", cond_box),
    "hand": ("손으로 들고", cond_hand),
    "shadow": ("짙은 그림자", cond_shadow),
    "fluorescent": ("형광등(파랑끼)", cond_fluorescent),
    "incandescent": ("백열등(노랑끼)", cond_incandescent),
    "motionblur": ("흔들림", cond_motionblur),
    "defocus": ("초점 안 맞음", cond_defocus),
    "jpeg": ("강한 JPEG 압축", cond_jpeg),
    "crop": ("일부 잘림", cond_crop),
}


# ── 측정 ──────────────────────────────────────────────────────────────
def topk_hits(logits: np.ndarray, truth_idx: int, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """행마다 top-1 / top-k 적중 여부."""
    order = np.argsort(-logits, axis=1)
    return order[:, 0] == truth_idx, (order[:, :k] == truth_idx).any(axis=1)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _self_test() -> None:
    """조건과 계산기를 일부러 깨뜨려 보고 믿는다."""
    # 단색 이미지로는 흐림 조건이 '아무것도 안 함'과 구별되지 않는다 — 무늬가 있어야 한다.
    yy, xx = np.mgrid[0:224, 0:224]
    tex = np.stack([(np.sin(xx / 5.0) * 110 + 130),
                    (np.cos(yy / 7.0) * 110 + 130),
                    ((xx ^ yy) % 256)], axis=-1)
    img = Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8))
    for key, (label, fn) in CONDITIONS.items():
        got = fn(img)
        if key == "studio":
            assert got.size == img.size
            continue
        # 총합은 흐림에서 거의 보존된다 — 화소별 최대 차이로 봐야 흐림을 잡는다
        a = np.asarray(got.resize((64, 64))).astype(int)
        b = np.asarray(img.resize((64, 64))).astype(int)
        assert got.size != img.size or np.abs(a - b).max() > 0, f"'{label}' 이 이미지를 안 바꾼다"
        assert E.to_tensor(got).shape == (1, 3, 224, 224)

    # 위 검사가 진짜 도는지 — 아무것도 안 하는 가짜 조건은 반드시 걸려야 한다
    try:
        got = (lambda im: im)(img)
        assert got.size != img.size or np.abs(
            np.asarray(got.resize((64, 64))).astype(int)
            - np.asarray(img.resize((64, 64))).astype(int)).max() > 0
    except AssertionError:
        pass
    else:
        raise AssertionError("검사가 헛돈다 — 아무것도 안 하는 조건을 통과시킨다")

    # top-k 계산기: 정답이 2등이면 top-1 은 틀리고 top-3 는 맞아야 한다
    # 동점을 두면 순위가 모호해져 검사가 무의미해진다 — 전부 다른 값으로.
    lg = np.array([[5.0, 9.0, 1.0, 0.5, 0.2],   # 정답 0 → 2등(top-3 안)
                   [0.0, 1.0, 2.0, 3.0, 9.0]])  # 정답 0 → 5등(top-3 밖)
    t1, t3 = topk_hits(lg, 0, k=3)
    assert list(t1) == [False, False], t1
    assert list(t3) == [True, False], t3

    # 음성 대조: 라벨을 한 칸씩 밀면 100% 가 나오면 안 된다
    lg2 = np.eye(5) * 9
    shifted = [topk_hits(lg2, (i + 1) % 5)[0][i] for i in range(5)]
    assert not any(shifted), "라벨을 틀리게 줘도 맞다고 한다 — 계산기가 헛돈다"

    print(f"조건 {len(CONDITIONS)}종 모두 이미지를 바꾸고, top-k 계산기는 오답을 오답이라 한다.")


def _sweep(session, picks, items, n: int) -> None:
    """세기를 바꿔가며 재는 이유 — '흔들리면 무너진다'는 말은 세기를 안 밝히면 의미가 없다.
    심하게 걸어놓고 무너졌다고 하면 과장이고, 어디서부터 무너지는지가 실제로 쓸 정보다."""
    grid = [("초점 흐림 반경", "px", [0, 1.0, 2.0, 3.0, 4.0, 6.0],
             lambda v: (lambda im: im if v == 0 else im.filter(ImageFilter.GaussianBlur(v)))),
            ("흔들림 길이", "%폭", [0, 2, 4, 7, 10],
             lambda v: (lambda im: im if v == 0 else _mb(im, v))),
            ("JPEG 품질", "q", [95, 80, 60, 40, 25, 12],
             lambda v: (lambda im: _jq(im, v)))]

    for title, unit, vals, mk in grid:
        print(f"\n[세기별] {title}")
        print(f"{unit:>8}  {'top-1':>7}{'top-3':>7}   " + "  ".join(f"{i:>4}" for i in items))
        for v in vals:
            fn = mk(v)
            t1s, t3s, cells = [], [], []
            for item, files in picks.items():
                lg = E.run(session, [E.to_tensor(fn(Image.open(f))) for f in files[:n]])
                a, b = topk_hits(lg, items.index(item))
                t1s.append(a.mean()); t3s.append(b.mean())
                cells.append(f"{a.mean():.0%}/{b.mean():.0%}")
            print(f"{v:>8}  {np.mean(t1s):>7.0%}{np.mean(t3s):>7.0%}   " + "  ".join(f"{c:>9}" for c in cells),
                  flush=True)


def _mb(im: Image.Image, pct: float) -> Image.Image:
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    k = max(2, int(im.size[0] * pct / 100))
    acc = np.zeros_like(a)
    for s in range(k):
        acc += np.roll(a, s - k // 2, axis=1)
    return Image.fromarray(np.clip(acc / k, 0, 255).astype(np.uint8))


def _jq(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=q, subsampling=2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="품목당 표본 수")
    ap.add_argument("--conditions", default="")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="흐림·압축 세기별 반응곡선")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return 0

    if args.sweep:
        s = ort.InferenceSession(str(E.MODEL), providers=["CPUExecutionProvider"])
        _sweep(s, E.sample_images(args.n), E.ITEMS, args.n)
        return 0

    session = ort.InferenceSession(str(E.MODEL), providers=["CPUExecutionProvider"])
    picks = E.sample_images(args.n)
    items = E.ITEMS
    keys = [k.strip() for k in args.conditions.split(",") if k.strip()] or list(CONDITIONS)
    results: dict[str, dict] = {}

    for key in keys:
        label, fn = CONDITIONS[key]
        per: dict[str, dict] = {}
        for item, files in picks.items():
            tensors = [E.to_tensor(fn(Image.open(f))) for f in files]
            logits = E.run(session, tensors)
            t1, t3 = topk_hits(logits, items.index(item))
            probs = softmax(logits)
            wrong = ~t1
            conf_wrong = float(probs[wrong].max(axis=1).mean()) if wrong.any() else 0.0
            mis = [items[i] for i in logits.argmax(axis=1)[wrong]]
            top_mis = max(set(mis), key=mis.count) if mis else ""
            per[item] = {
                "top1": float(t1.mean()), "top3": float(t3.mean()),
                "n": len(files), "top_mistake": top_mis,
                "mistake_share": float(mis.count(top_mis) / len(mis)) if mis else 0.0,
                "conf_when_wrong": conf_wrong,
            }
        results[key] = {
            "label": label,
            "top1": float(np.mean([v["top1"] for v in per.values()])),
            "top3": float(np.mean([v["top3"] for v in per.values()])),
            "items": per,
        }
        r = results[key]
        print(f"[측정] {label:<20} top1 {r['top1']:6.1%}  top3 {r['top3']:6.1%}", flush=True)

    base1 = results["studio"]["top1"] if "studio" in results else 1.0
    base3 = results["studio"]["top3"] if "studio" in results else 1.0

    order = sorted((k for k in results if k != "studio"), key=lambda k: results[k]["top3"])
    w = max(len(v["label"]) for v in results.values()) + 2

    print(f"\n품목당 {args.n}장 x 5품목 = {args.n * 5}장/조건 · 브라우저 전처리 재현")
    print("(품목 5종이므로 top-3 무작위 기대값은 60% 다)\n")
    head = f"{'조건':<{w}}{'top-1':>7}{'top-3':>8}{'Δtop-1':>9}{'Δtop-3':>9}   " + \
           "  ".join(f"{i:>4}" for i in items) + "   (칸=품목별 top-1/top-3)"
    print(head)
    print("-" * 108)

    def line(key: str) -> None:
        r = results[key]
        cells = "  ".join(f"{r['items'][i]['top1']:.0%}/{r['items'][i]['top3']:.0%}" for i in items)
        print(f"{r['label']:<{w}}{r['top1']:>7.0%}{r['top3']:>8.0%}"
              f"{r['top1'] - base1:>+9.0%}{r['top3'] - base3:>+9.0%}   {cells}")

    if "studio" in results:
        line("studio")
        print("-" * 108)
    for key in order:
        line(key)

    print("\n가장 위험한 조건(top-3 낮은 순):")
    for key in order[:3]:
        r = results[key]
        bad = sorted(r["items"].items(), key=lambda kv: kv[1]["top3"])[:2]
        detail = ", ".join(
            f"{it}→'{v['top_mistake']}' {v['mistake_share']:.0%}(신뢰도 {v['conf_when_wrong']:.0%})"
            for it, v in bad if v["top_mistake"])
        print(f"  {r['label']}: top-1 {r['top1']:.0%} / top-3 {r['top3']:.0%}   {detail}")

    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
