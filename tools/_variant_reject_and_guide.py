"""인식하지 말아야 할 때를 아는 게이트 — 차지비율·선명도로 거르고, 잘라서 되살린다.

문제: 피사체가 작아지면 top-1 이 100%→43% 로 무너지고, 감자는 멀어지면 29/30 장이
'사과'가 된다(평균 신뢰도 54%). 화면은 후보 3개를 보여주는데 감자 far 는 top-3 에도 없다.
틀린 답을 자신 있게 내놓는 것보다 "다시 찍어주세요"가 낫다.

여기서 만드는 것은 서버가 받은 224x224 RGB 만으로 계산하는 값싼 지표 두 개다.

    fill   피사체가 프레임의 몇 %를 차지하는가(선형 비율 = sqrt(bbox 면적비))
    sharp  초점이 맞았는가 (var(Laplacian) / var(gray) — 대비로 정규화해 어두워도 안 흔들린다)

fill 은 두 방식의 max 다. 하나만 쓰면 각각 반대쪽에서 틀린다.
    col  코너 medoid 로 배경색을 잡고 그 색과 다른 칸의 bbox  → 배경이 균일할 때 정확
    det  블록 내 표준편차(디테일)가 있는 칸의 bbox           → 배경색을 안 쓴다
피사체가 프레임을 꽉 채우면 코너까지 피사체라 col 이 무너진다(근접촬영 col p5=0.51).
그때 det=1.00 이 받쳐 준다. 반대로 흙배경처럼 색이 다른 곳에서는 col 이 받쳐 준다.

그리고 이 지표는 bbox 를 이미 알고 있으므로, 거부하는 대신 그 자리를 잘라
한 번 더 추론할 수 있다(추론 1회 추가, 클라이언트 변경 없음).

    python tools/_variant_reject_and_guide.py --self-test   # 지표부터 검증
    python tools/_variant_reject_and_guide.py               # 본 측정
    python tools/_variant_reject_and_guide.py --only sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_realworld as E  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ITEMS = E.ITEMS
SIZE = 224
GRID = 28                 # 224 = 28 x 8 — 8x8 블록 평균으로 고주파 노이즈를 죽인다
CELL = SIZE // GRID
GRAY = (150, 148, 143)    # eval_realworld.cond_far 가 쓰는 배경색

BG_TOL = 20.0             # 코너 배경색과의 RGB 거리 문턱
DET_REL = 0.12            # 디테일 상대 문턱 (p95 대비)
DET_ABS = 1.0             # 디테일 절대 하한 (8bit 계조)

# 아래 두 값은 calibrate() 가 train 분할에서 정한 값이다(valid 는 안 봤다).
# 규칙: t_fill = 좋은사진 fill 1퍼센타일 − 0.05,  t_sharp = 좋은사진 sharp 1퍼센타일 ÷ 2
T_FILL = 0.736            # 이보다 작으면 '너무 멀다' → 잘라서 다시 본다
T_SHARP = 0.00218         # 이보다 낮으면 '흔들렸다' → 다시 찍게 한다

_SESSION: ort.InferenceSession | None = None


def session() -> ort.InferenceSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = ort.InferenceSession(str(E.MODEL), providers=["CPUExecutionProvider"])
    return _SESSION


# ══ 지표 ═══════════════════════════════════════════════════════════════
def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """마스크의 경계상자(격자 단위). 칸이 3개 미만이면 '피사체 없음'."""
    if mask.sum() < 3:
        return None
    ys, xs = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _lin(box: tuple[int, int, int, int] | None) -> float:
    """bbox 를 선형 차지비율로. (_diag_fill 의 linear_pct 와 같은 축)"""
    if box is None:
        return 0.0
    x0, y0, x1, y1 = box
    return float(np.sqrt((x1 - x0) * (y1 - y0) / (GRID * GRID)))


def quality(img: Image.Image) -> dict:
    """서버가 받는 224x224 RGB 하나에서 fill·sharp·bbox 를 뽑는다.

    numpy 만 쓰고 224x224 한 장에 대해 상수 시간이다(모델 추론 전에 돈다).
    """
    a = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
    gray = a.mean(axis=2)

    # 8x8 블록 평균(색)과 블록 내 표준편차(디테일)
    coarse = a.reshape(GRID, CELL, GRID, CELL, 3).mean(axis=(1, 3))
    detail = gray.reshape(GRID, CELL, GRID, CELL).std(axis=(1, 3))

    # ── col: 코너 medoid 로 배경색 추정 → 그 색과 다른 칸
    k = 3
    meds = np.stack([np.median(p.reshape(-1, 3), axis=0) for p in
                     (coarse[:k, :k], coarse[:k, -k:], coarse[-k:, :k], coarse[-k:, -k:])])
    dist = np.sqrt(((meds[:, None, :] - meds[None, :, :]) ** 2).sum(axis=2))
    agree = (dist <= BG_TOL).sum(axis=1)
    best = int(agree.argmax())
    bg = meds[dist[best] <= BG_TOL].mean(axis=0)
    m_col = np.sqrt(((coarse - bg) ** 2).sum(axis=2)) > BG_TOL

    # ── det: 배경색을 쓰지 않는 쪽. 평평한 배경은 표준편차가 0 에 가깝다
    m_det = detail > max(DET_REL * float(np.percentile(detail, 95)), DET_ABS)

    b_col, b_det = _bbox(m_col), _bbox(m_det)
    f_col, f_det = _lin(b_col), _lin(b_det)
    # 둘 중 큰 쪽을 믿는다 — 오거부(좋은 사진을 내치는 것)가 더 비싸다
    box = b_col if f_col >= f_det else b_det

    # ── sharp: 대비로 정규화한 라플라시안 분산
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    sharp = float(lap.var() / (gray.var() + 1e-6))

    return {
        "fill": max(f_col, f_det),
        "fill_col": f_col,
        "fill_det": f_det,
        "sharp": sharp,
        "box": box,
        "bg_corners": int(agree[best]),
        "found": box is not None,
        "arr": a,
    }


def decide(q: dict, t_fill: float = T_FILL, t_sharp: float = T_SHARP) -> str:
    """판정할지, 잘라서 판정할지, 다시 찍게 할지.

    '피사체 없음'을 먼저 본다 — 빈 화면은 선명도도 0 이라 순서를 바꾸면 "흔들렸어요"라는
    엉뚱한 안내가 나간다. 그 다음이 선명도인 이유는 흐린 사진은 잘라도 못 살리기 때문이다.
    """
    if not q["found"]:
        return "reject_empty"
    if q["sharp"] < t_sharp:
        return "reject_blur"
    if q["fill"] < t_fill:
        return "crop"
    return "ok"


GUIDE = {
    "ok": "",
    "crop": "조금 멀어요 — 잘라서 봤습니다. 더 가까이 찍으면 정확해집니다.",
    "reject_blur": "흔들렸어요. 폰을 고정하고 다시 찍어주세요.",
    "reject_empty": "무엇을 찍었는지 모르겠어요. 농산물이 화면 가운데 오게 찍어주세요.",
}


def cropped(q: dict, pad: int = 2) -> Image.Image:
    """이미 구한 bbox 로 잘라 224 로 되돌린다 — 추가 비용은 추론 1회뿐."""
    im = Image.fromarray(q["arr"].astype(np.uint8))
    if q["box"] is None:
        return im
    x0, y0, x1, y1 = q["box"]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(GRID, x1 + pad), min(GRID, y1 + pad)
    return im.crop((x0 * CELL, y0 * CELL, x1 * CELL, y1 * CELL)).resize((SIZE, SIZE), Image.LANCZOS)


# ══ 조건 ═══════════════════════════════════════════════════════════════
def compose(img: Image.Image, bg: tuple[int, int, int], occ: float) -> Image.Image:
    """_diag_potato.compose 와 같다 — 피사체가 프레임의 occ(선형)만 차지하게."""
    if occ >= 0.999:
        return img
    w, h = img.size
    cv = Image.new("RGB", (round(w / occ), round(h / occ)), bg)
    cv.paste(img, ((cv.width - w) // 2, (cv.height - h) // 2))
    return cv


def blur(img: Image.Image, px: float) -> Image.Image:
    """224 프레임 기준 px 만큼 흐리게(원본 해상도에 비례해서 건다)."""
    return img.filter(ImageFilter.GaussianBlur(radius=px * img.size[0] / SIZE))


def closeup(img: Image.Image, z: float) -> Image.Image:
    """피사체가 프레임을 완전히 덮는 이상적 근접촬영 — 오거부 시험용."""
    w, h = img.size
    s = int(min(w, h) / z)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((w, h), Image.LANCZOS)


def clutter(img: Image.Image, occ: float, blob: int = 40) -> Image.Image:
    """큰 얼룩이 깔린 배경 위의 작은 피사체 — 이 지표의 적대 조건."""
    w, h = img.size
    W, H = round(w / occ), round(h / occ)
    rng = np.random.default_rng(0)
    small = rng.integers(70, 170, (max(2, H // blob), max(2, W // blob), 3)).astype(np.uint8)
    cv = Image.fromarray(small).resize((W, H), Image.BICUBIC)
    cv.paste(img, ((W - w) // 2, (H - h) // 2))
    return cv


GOOD = {   # 이미 100% 인 조건들 — 여기서 거부되면 그게 '답답함'이다
    "studio": lambda im: im,
    "stretch43": E.cond_stretch43,
    "portrait34": E.cond_portrait34,
    "background": E.cond_background,
    "dim": E.cond_dim,
    "tilt": E.cond_tilt,
    "closeup": lambda im: closeup(im, 2.6),
}
BAD = {    # 무너지는 조건들 — 여기서 거부되는 것은 이득이다
    "far": E.cond_far,
    "fill50": lambda im: compose(im, GRAY, 0.50),
    "fill35": lambda im: compose(im, GRAY, 0.35),
    "fill25": lambda im: compose(im, GRAY, 0.25),
    "blur2": lambda im: blur(im, 2.0),
    "blur4": lambda im: blur(im, 4.0),
    "blur8": lambda im: blur(im, 8.0),
}
HOSTILE = {  # 지표가 못 잡는다고 미리 밝히는 곳
    "clutter25": lambda im: clutter(im, 0.25),
    "clutter50": lambda im: clutter(im, 0.50),
}
ALL_CONDS = {**GOOD, **BAD, **HOSTILE}


# ══ 검증 ═══════════════════════════════════════════════════════════════
def _synth(side: int, bg=(255, 255, 255), fg=(190, 70, 45)) -> Image.Image:
    im = Image.new("RGB", (SIZE, SIZE), bg)
    o = (SIZE - side) // 2
    d = ImageDraw.Draw(im)
    d.rectangle([o, o, o + side - 1, o + side - 1], fill=fg)
    # 단색 사각형은 디테일이 0 이라 det 이 못 본다 — 실제 농산물처럼 무늬를 넣는다
    for i in range(o, o + side, 6):
        d.line([i, o, i, o + side - 1], fill=(fg[0] - 45, fg[1] - 25, fg[2] - 20))
    return im


def _self_test() -> None:
    """아는 답을 맞히는지. 하나라도 틀리면 아래 측정은 전부 의미가 없다."""
    n = 0

    # 1) 아는 크기 — fill 은 선형 비율이어야 한다 (격자 28칸 → ±1칸 = ±0.04)
    for side, want in [(224, 1.00), (168, 0.75), (112, 0.50), (56, 0.25)]:
        got = quality(_synth(side))["fill"]
        assert abs(got - want) <= 0.09, f"{side}px 사각형: fill={got:.2f}, 기대 {want:.2f}"
        n += 1

    # 2) 빈 프레임 — 피사체가 없으면 없다고 해야 한다(0% 로 조용히 통과하면 안 된다).
    #    '거부됐다'만 보면 선명도 문턱이 대신 잡아줘도 통과해 버리므로 사유까지 못박는다.
    q = quality(Image.new("RGB", (SIZE, SIZE), (245, 245, 245)))
    assert not q["found"], f"빈 화면에서 피사체를 찾았다: fill={q['fill']:.2f}"
    assert decide(q) == "reject_empty", f"빈 화면 사유가 틀렸다: {decide(q)}"
    n += 2

    # 3) 흐림 — 같은 그림을 흐리게 하면 sharp 만 떨어지고 fill 은 유지돼야 한다
    base = _synth(200)
    qs, qb = quality(base), quality(blur(base, 4.0))
    assert qb["sharp"] < qs["sharp"] / 10, f"흐리게 해도 sharp 가 안 떨어진다: {qs['sharp']:.4f}→{qb['sharp']:.4f}"
    assert qb["fill"] > 0.8, f"흐림이 fill 을 망가뜨린다: {qb['fill']:.2f}"
    assert decide(qb) == "reject_blur" and decide(qs) == "ok", (decide(qs), decide(qb))
    n += 2

    # 4) 어두움이 선명도를 흉내내면 안 된다 — 대비 정규화가 실제로 되는지
    dim = E.cond_dim(base)
    assert quality(dim)["sharp"] > T_SHARP * 3, f"어두운 사진을 흔들렸다고 본다: {quality(dim)['sharp']:.4f}"
    n += 1

    # 5) 실제 학습 이미지: 원본은 통과, 같은 것을 25% 로 줄이면 crop 판정
    f0 = next(iter(E.sample_images(1).values()))[0]
    im = Image.open(f0).convert("RGB")
    assert decide(quality(im)) == "ok", f"학습과 같은 사진을 거부했다: {quality(im)}"
    q25 = quality(compose(im, GRAY, 0.25))
    assert decide(q25) == "crop", f"25% 짜리를 통과시켰다: fill={q25['fill']:.2f}"
    assert abs(q25["fill"] - 0.25) < 0.10, f"fill 이 occ 축과 어긋난다: {q25['fill']:.2f}"
    n += 3

    # 6) 근접촬영 오거부 방지 — 코너까지 피사체인 사진에서 col 은 무너져도 det 이 받쳐야 한다
    qc = quality(closeup(im, 2.6))
    assert decide(qc) == "ok", f"이상적 근접촬영을 거부했다: col={qc['fill_col']:.2f} det={qc['fill_det']:.2f}"
    n += 1

    print(f"✅ 자체검증 {n}건 통과 — 아는 크기(±0.09), 빈 화면, 흐림, 어두움, occ 축, 근접촬영")


def _fail_test() -> None:
    """검사가 헛돌지 않는지 — 일부러 망가뜨린 지표는 반드시 걸려야 한다."""
    global DET_REL, BG_TOL
    keep = (DET_REL, BG_TOL)
    try:
        DET_REL, BG_TOL = 10.0, 5000.0     # 어떤 칸도 피사체가 아니게 만든다
        q = quality(_synth(168))
        assert not q["found"] and decide(q) == "reject_empty", f"망가뜨렸는데 통과한다: {q['fill']:.2f}"
        print("✅ 역검증 — 문턱을 망가뜨리자 '피사체 없음'으로 바뀐다(검사가 살아 있다)")
    finally:
        DET_REL, BG_TOL = keep

    # 선명도 검사도 헛돌지 않는지: 문턱을 0 으로 두면 흐린 사진이 통과해야 한다
    q = quality(blur(_synth(200), 4.0))
    assert decide(q, t_sharp=0.0) != "reject_blur", "sharp 문턱이 아무 일도 안 한다"
    print("✅ 역검증 — sharp 문턱을 0 으로 두면 흐린 사진이 통과한다(문턱이 실제로 작동)")


# ══ 문턱 정하기 — train 에서만 ═════════════════════════════════════════
def calibrate(n: int = 30, seed: int = 11) -> tuple[float, float, dict]:
    """문턱을 train 분할에서 정한다. valid(=측정 표본)는 건드리지 않는다.

    규칙을 먼저 못박고 그대로 따른다 — 결과를 보고 고르면 그건 측정이 아니다.
        t_sharp = 좋은 사진 sharp 1퍼센타일 ÷ 2      (2배 여유)
        t_fill  = 좋은 사진 fill  1퍼센타일 − 0.05   (여유 0.05)
    둘 다 '좋은 사진을 거의 안 내치는' 쪽으로만 정의돼 있다. 나쁜 사진 쪽 수치는
    보지 않는다 — 보면 그 순간 문턱이 정답을 훔친 것이 된다.
    모델을 안 돌리므로(quality 만 쓴다) 이 단계는 사실상 공짜다.
    """
    import random
    train = E.ROOT / "data" / "onfarm_cv" / "train"
    rng = random.Random(seed)
    fills, sharps = [], []
    for item in ITEMS:
        base = train / item
        if not base.exists():
            continue
        files = [p for g in sorted(base.iterdir()) if g.is_dir() for p in g.glob("*.jpg")]
        rng.shuffle(files)
        for f in files[:n]:
            im = Image.open(f).convert("RGB")
            for fn in GOOD.values():
                q = quality(fn(im))
                fills.append(q["fill"]); sharps.append(q["sharp"])
    fp1 = float(np.percentile(fills, 1))
    sp1 = float(np.percentile(sharps, 1))
    return (round(max(0.30, fp1 - 0.05), 3), round(sp1 / 2, 5),
            {"n": len(fills), "fill_p1": fp1, "sharp_p1": sp1,
             "fill_med": float(np.median(fills)), "sharp_med": float(np.median(sharps))})


# ══ 측정 ═══════════════════════════════════════════════════════════════
def _logits(imgs: list[Image.Image]) -> np.ndarray:
    ses = session()
    x = np.concatenate([E.to_tensor(im) for im in imgs], axis=0)
    out = ses.run(None, {ses.get_inputs()[0].name: x})
    names = [o.name for o in ses.get_outputs()]
    return np.asarray(out[names.index("item_logits")])


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def collect(picks: dict[str, list[Path]], conds: dict) -> dict:
    """조건×품목마다 한 번씩만 돌고 전부 캐시한다 — 이후 표는 이 캐시에서 나온다."""
    cache: dict = {}
    for cname, fn in conds.items():
        for item, files in picks.items():
            imgs = [fn(Image.open(f).convert("RGB")) for f in files]
            qs = [quality(im) for im in imgs]
            lg0 = _logits(imgs)
            lg1 = _logits([cropped(q) for q in qs])
            ti = ITEMS.index(item)
            r0, r1 = np.argsort(-lg0, 1), np.argsort(-lg1, 1)
            cache[(cname, item)] = {
                "fill": np.array([q["fill"] for q in qs]),
                "sharp": np.array([q["sharp"] for q in qs]),
                "found": np.array([q["found"] for q in qs]),
                "t1": (r0[:, 0] == ti), "t3": (r0[:, :3] == ti).any(1),
                "c1": (r1[:, 0] == ti), "c3": (r1[:, :3] == ti).any(1),
                "conf": _softmax(lg0).max(1),
            }
    return cache


def _gate(rec: dict, tf: float, ts: float) -> dict[str, np.ndarray]:
    blurry = rec["sharp"] < ts
    empty = ~rec["found"] & ~blurry
    small = ~blurry & ~empty & (rec["fill"] < tf)
    return {"reject": blurry | empty, "crop": small,
            "ok": ~(blurry | empty | small)}


def _score(recs: list[dict], tf: float, ts: float, use_crop: bool) -> dict:
    """거부율과, '답한 사진들' 기준 top-1/top-3."""
    n = t1 = t3 = rej = ans = 0
    for r in recs:
        g = _gate(r, tf, ts)
        n += len(r["t1"])
        rej += int(g["reject"].sum())
        if use_crop:
            keep_ok, keep_cr = g["ok"], g["crop"]
            ans += int(keep_ok.sum() + keep_cr.sum())
            t1 += int(r["t1"][keep_ok].sum() + r["c1"][keep_cr].sum())
            t3 += int(r["t3"][keep_ok].sum() + r["c3"][keep_cr].sum())
        else:
            rej += int(g["crop"].sum())          # crop 을 안 쓰면 그냥 거부
            keep = g["ok"]
            ans += int(keep.sum())
            t1 += int(r["t1"][keep].sum()); t3 += int(r["t3"][keep].sum())
    return {"n": n, "reject_pct": rej / n, "answered": ans,
            "t1": t1 / ans if ans else float("nan"),
            "t3": t3 / ans if ans else float("nan")}


def _base(recs: list[dict]) -> tuple[float, float, int]:
    n = sum(len(r["t1"]) for r in recs)
    return (sum(int(r["t1"].sum()) for r in recs) / n,
            sum(int(r["t3"].sum()) for r in recs) / n, n)


def table_conditions(cache: dict, tf: float, ts: float) -> None:
    print("\n" + "=" * 100)
    print(f"1. 조건별 — 게이트 없음 vs 게이트(거부만) vs 게이트+크롭   (fill<{tf:.2f} 또는 sharp<{ts:.3f})")
    print("=" * 100)
    hdr = (f"{'조건':<12}{'n':>4}{'게이트없음':>12}  |{'거부%':>7}{'통과 top1':>10}{'top3':>7}"
           f"  |{'거부%':>7}{'응답 top1':>10}{'top3':>7}   {'fill 중앙':>9}{'sharp 중앙':>11}")
    print(hdr)
    print("-" * len(hdr))
    for group, conds in (("좋은 사진", GOOD), ("무너지는 사진", BAD), ("적대 조건", HOSTILE)):
        print(f"[{group}]")
        for cname in conds:
            recs = [cache[(cname, it)] for it in ITEMS if (cname, it) in cache]
            b1, b3, n = _base(recs)
            a = _score(recs, tf, ts, use_crop=False)
            c = _score(recs, tf, ts, use_crop=True)
            fill = np.median(np.concatenate([r["fill"] for r in recs]))
            sh = np.median(np.concatenate([r["sharp"] for r in recs]))
            f1 = f"{a['t1']:>9.0%}" if a["answered"] else f"{'—':>9}"
            f3 = f"{a['t3']:>7.0%}" if a["answered"] else f"{'—':>7}"
            print(f"{cname:<12}{n:>4}{b1:>6.0%}/{b3:<5.0%}  |{a['reject_pct']:>7.0%}{f1}{f3}"
                  f"  |{c['reject_pct']:>7.0%}{c['t1']:>10.0%}{c['t3']:>7.0%}"
                  f"   {fill:>9.2f}{sh:>11.4f}")


def table_sweep(cache: dict, tf0: float, ts0: float) -> None:
    print("\n" + "=" * 100)
    print("2. 트레이드오프 — fill 문턱을 올리면 정확도는 오르고 거부율도 오른다")
    print("   '좋은 사진 거부율'이 답답함의 값이다(20% 넘으면 시연에서 못 쓴다).")
    print("=" * 100)
    good = [cache[(c, it)] for c in GOOD for it in ITEMS if (c, it) in cache]
    bad = [cache[(c, it)] for c in BAD for it in ITEMS if (c, it) in cache]
    allr = good + bad
    gb1, gb3, gn = _base(good)
    bb1, bb3, bn = _base(bad)
    ab1, ab3, an = _base(allr)
    print(f"\n  게이트 없음: 좋은 사진 {gn}장 {gb1:.0%}/{gb3:.0%}   "
          f"무너지는 사진 {bn}장 {bb1:.0%}/{bb3:.0%}   합계 {an}장 {ab1:.0%}/{ab3:.0%}")

    for use_crop, title in ((False, "A. 거부만 한다 (문턱 미달 = 다시 찍기)"),
                            (True, "B. 거부 + 크롭 재추론 (작으면 잘라서 답한다)")):
        print(f"\n  {title}")
        hdr = (f"    {'fill문턱':>8}{'sharp문턱':>10} |{'좋은사진 거부%':>15}{'무너진사진 거부%':>17}"
               f" |{'전체 거부%':>11}{'응답 top1':>10}{'top3':>7}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for tf in sorted({0.00, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, tf0, 0.80, 0.90}):
            ts = ts0
            g = _score(good, tf, ts, use_crop)
            b = _score(bad, tf, ts, use_crop)
            a = _score(allr, tf, ts, use_crop)
            mark = "  ←채택" if (abs(tf - tf0) < 1e-9 and use_crop) else ""
            print(f"    {tf:>8.2f}{ts:>10.4f} |{g['reject_pct']:>15.0%}{b['reject_pct']:>17.0%}"
                  f" |{a['reject_pct']:>11.0%}{a['t1']:>10.0%}{a['t3']:>7.0%}{mark}")

    print("\n  sharp 문턱만 따로 (fill 문턱은 0 으로 두어 크기 효과를 뺀다)")
    hdr = (f"    {'sharp문턱':>10} |{'좋은사진 거부%':>15}{'무너진사진 거부%':>17}"
           f" |{'전체 거부%':>11}{'응답 top1':>10}{'top3':>7}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for ts in sorted({0.0, 0.001, 0.002, ts0, 0.004, 0.006, 0.010, 0.020}):
        g = _score(good, 0.0, ts, False)
        b = _score(bad, 0.0, ts, False)
        a = _score(allr, 0.0, ts, False)
        mark = "  ←채택" if abs(ts - ts0) < 1e-12 else ""
        print(f"    {ts:>10.4f} |{g['reject_pct']:>15.0%}{b['reject_pct']:>17.0%}"
              f" |{a['reject_pct']:>11.0%}{a['t1']:>10.0%}{a['t3']:>7.0%}{mark}")

    print("\n  거부는 전부 '흐림'에서 나온다(작으면 잘라서 답한다). 그래서 전체 거부율은")
    print("  풀에 흐린 사진이 몇 % 섞였느냐로 정해진다:  거부율 ≈ 흐린비율×100% + 나머지×1%")
    print("  위 표의 '전체 거부율'은 일부러 절반을 망친 사진으로 채운 풀 기준이라 상한에 가깝다.")


def table_potato(cache: dict, tf: float, ts: float) -> None:
    print("\n" + "=" * 100)
    print("3. 품목별 — studio / far / 차지비율 50·35·25%  (감자 far 가 지금 0/0 인 지점)")
    print("=" * 100)
    hdr = (f"{'조건':<10}{'품목':<6}{'게이트없음 t1/t3':>18}{'거부%':>8}{'크롭%':>8}"
           f"{'응답 t1':>9}{'t3':>7}   비고")
    print(hdr)
    print("-" * len(hdr))
    for cname in ("studio", "far", "fill50", "fill35", "fill25"):
        for item in ITEMS:
            r = cache[(cname, item)]
            g = _gate(r, tf, ts)
            b1, b3, n = _base([r])
            s = _score([r], tf, ts, use_crop=True)
            note = ""
            if item == "감자" and cname == "far":
                note = "← 원래 29/30 장이 '사과'"
            print(f"{cname:<10}{item:<6}{b1:>10.0%}/{b3:<7.0%}{g['reject'].mean():>8.0%}"
                  f"{g['crop'].mean():>8.0%}{s['t1']:>9.0%}{s['t3']:>7.0%}   {note}")
        print()

    r = cache[("far", "감자")]
    g = _gate(r, tf, ts)
    print(f"  감자 far 요약 (n={len(r['t1'])}):")
    print(f"    게이트 없음      : top-1 {r['t1'].mean():.0%}  top-3 {r['t3'].mean():.0%}  "
          f"— 후보 3개에 감자가 아예 없다")
    print(f"    거부만           : {g['reject'].sum() + g['crop'].sum()}/{len(r['t1'])} 장 '다시 찍기'  "
          f"→ 틀린 답을 내놓지는 않지만 답도 못 준다")
    print(f"    크롭 재추론      : top-1 {r['c1'][g['crop']].mean() if g['crop'].any() else float('nan'):.0%}  "
          f"top-3 {r['c3'][g['crop']].mean() if g['crop'].any() else float('nan'):.0%}  "
          f"(크롭 대상 {g['crop'].sum()}장 기준)")


def table_confidence(cache: dict, tf: float, ts: float) -> None:
    print("\n" + "=" * 100)
    print("4. 대조군 — 그냥 모델 신뢰도로 거르면 되는 것 아닌가?")
    print("   (신뢰도는 추론을 해야 나오고 '무엇을 고치라'고 말해주지 못한다. 성능만 비교한다.)")
    print("=" * 100)
    good = [cache[(c, it)] for c in GOOD for it in ITEMS if (c, it) in cache]
    bad = [cache[(c, it)] for c in BAD for it in ITEMS if (c, it) in cache]
    hos = [cache[(c, it)] for c in HOSTILE for it in ITEMS if (c, it) in cache]

    def conf_score(recs, thr):
        n = t1 = t3 = ans = 0
        for r in recs:
            k = r["conf"] >= thr
            n += len(k); ans += int(k.sum())
            t1 += int(r["t1"][k].sum()); t3 += int(r["t3"][k].sum())
        return {"reject_pct": 1 - ans / n, "t1": t1 / ans if ans else float("nan"),
                "t3": t3 / ans if ans else float("nan")}

    hdr = (f"    {'신뢰도문턱':>11} |{'좋은사진 거부%':>15}{'무너진사진 거부%':>17}"
           f"{'적대조건 거부%':>16} |{'전체 응답 top1':>15}{'top3':>7}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        g, b, h = conf_score(good, thr), conf_score(bad, thr), conf_score(hos, thr)
        a = conf_score(good + bad, thr)
        print(f"    {thr:>11.0%} |{g['reject_pct']:>15.0%}{b['reject_pct']:>17.0%}"
              f"{h['reject_pct']:>16.0%} |{a['t1']:>15.0%}{a['t3']:>7.0%}")

    hg = _score(hos, tf, ts, use_crop=True)
    hb1, hb3, _ = _base(hos)
    print(f"\n    화상지표 게이트(fill<{tf}, sharp<{ts})가 적대조건에서: 거부 {hg['reject_pct']:.0%}, "
          f"응답 {hg['t1']:.0%}/{hg['t3']:.0%} (게이트 없음 {hb1:.0%}/{hb3:.0%})")
    print("    → 어질러진 배경은 화상지표가 못 잡는다. 신뢰도 문턱이 여기서는 낫다.")


def table_combined(cache: dict, tf: float, ts: float) -> None:
    print("\n" + "=" * 100)
    print("5. 합친 것 — 화상지표(크롭 포함) 다음에 신뢰도 문턱을 한 번 더")
    print("=" * 100)
    good = [cache[(c, it)] for c in GOOD for it in ITEMS if (c, it) in cache]
    bad = [cache[(c, it)] for c in BAD for it in ITEMS if (c, it) in cache]
    hos = [cache[(c, it)] for c in HOSTILE for it in ITEMS if (c, it) in cache]
    hdr = (f"    {'신뢰도문턱':>11} |{'좋은사진 거부%':>15}{'무너진사진 거부%':>17}{'적대조건 거부%':>16}"
           f" |{'좋+무너 응답 top1':>18}{'top3':>7}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for cthr in (0.0, 0.4, 0.5, 0.6, 0.7):
        cells = []
        for recs in (good, bad, hos, good + bad):
            n = t1 = t3 = ans = 0
            for r in recs:
                g = _gate(r, tf, ts)
                # crop 대상은 크롭 결과의 신뢰도를 다시 재야 정확하지만,
                # 여기서는 크롭 전 신뢰도로 거른 뒤 크롭 결과로 답한다(보수적).
                keep_ok = g["ok"] & (r["conf"] >= cthr)
                keep_cr = g["crop"]
                n += len(r["t1"]); ans += int(keep_ok.sum() + keep_cr.sum())
                t1 += int(r["t1"][keep_ok].sum() + r["c1"][keep_cr].sum())
                t3 += int(r["t3"][keep_ok].sum() + r["c3"][keep_cr].sum())
            cells.append((1 - ans / n, t1 / ans if ans else float("nan"),
                          t3 / ans if ans else float("nan")))
        print(f"    {cthr:>11.0%} |{cells[0][0]:>15.0%}{cells[1][0]:>17.0%}{cells[2][0]:>16.0%}"
              f" |{cells[3][1]:>18.0%}{cells[3][2]:>7.0%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="품목당 표본 수(baseline 과 같게)")
    ap.add_argument("--fill", type=float, default=None, help="생략하면 train 에서 정한다")
    ap.add_argument("--sharp", type=float, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    _self_test()
    _fail_test()
    if args.self_test:
        return 0

    if args.fill is None or args.sharp is None:
        tf, ts, info = calibrate(args.n)
        args.fill = tf if args.fill is None else args.fill
        args.sharp = ts if args.sharp is None else args.sharp
        print(f"\n문턱 보정(train 분할 {info['n']}장, 좋은 조건만, 모델 미사용):")
        print(f"  fill  중앙 {info['fill_med']:.2f}  1퍼센타일 {info['fill_p1']:.2f}  → 문턱 {args.fill:.3f}")
        print(f"  sharp 중앙 {info['sharp_med']:.4f}  1퍼센타일 {info['sharp_p1']:.4f}  → 문턱 {args.sharp:.5f}")
    else:
        print(f"\n문턱 수동 지정: fill<{args.fill} sharp<{args.sharp}")

    picks = E.sample_images(args.n)          # seed=7 고정 — baseline 과 같은 표본
    got = {k: len(v) for k, v in picks.items()}
    print(f"\n표본: valid 분할, 품목당 {args.n}장 {got}  (E.sample_images(seed=7))")
    cache = collect(picks, ALL_CONDS)

    want = [s.strip() for s in args.only.split(",") if s.strip()]
    run = lambda k: (not want) or k in want  # noqa: E731
    if run("cond"):
        table_conditions(cache, args.fill, args.sharp)
    if run("sweep"):
        table_sweep(cache, args.fill, args.sharp)
    if run("potato"):
        table_potato(cache, args.fill, args.sharp)
    if run("conf"):
        table_confidence(cache, args.fill, args.sharp)
    if run("combined"):
        table_combined(cache, args.fill, args.sharp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
