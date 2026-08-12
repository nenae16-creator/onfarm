"""피사체를 찾아 자동으로 확대하는 전처리 — 구현과 측정(scratch).

왜: 이 모델은 피사체가 프레임을 꽉 채운 사진(학습 분포에서 선형 차지비율 중앙값 97~100%)
으로만 학습됐다. 그래서 멀리서 찍어 피사체가 작아지면 top-1 이 무너진다(감자 far 는 0%).
검출 모델을 새로 얹지 않고, 배경색 거리 + 엣지 밀도 + 중앙 가중으로 관심영역을 잡아
잘라낸 뒤 224 로 만들면 얼마나 되돌아오는지를 잰다.

    python tools/_variant_saliency_crop.py --self-test    # 측정기·크롭기 먼저 검증
    python tools/_variant_saliency_crop.py                # 본 측정(baseline 대조)
    python tools/_variant_saliency_crop.py --ablation     # 여백/구성요소 민감도

두 가지 배치 위치를 따로 잰다 — 같은 알고리즘이지만 입력 해상도가 다르다:
    client : 브라우저가 원본 프레임에서 자른 뒤 224 로 줄인다(features.js 안, 새 패키지 0).
    server : 서버가 이미 224 로 뭉개진 이미지에서 자른다(클라이언트 무수정, 대신 화질 손실).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_realworld as E  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ITEMS = E.ITEMS
SESSION = ort.InferenceSession(str(E.MODEL), providers=["CPUExecutionProvider"])

# ── 조정값 (모두 근거를 적는다) ────────────────────────────────────────
WORK = 128          # 관심영역을 찾는 작업 해상도. 224 원본에서도 subject 가 충분히 크다.
CORNER = 0.10       # 배경색 추정용 네 모서리 패치 크기(변 길이 대비) — _diag_fill 과 같은 발상
W_COLOR = 0.65      # saliency = 색거리 0.65 + 엣지 0.35
W_EDGE = 0.35
CENTER_FLOOR = 0.55  # 중앙 가중: 가장자리도 0.55 배는 남긴다(피사체가 중앙에만 있다고 못 박지 않게)
CENTER_SIGMA = 0.55
TARGET_FILL = 0.95  # 자른 뒤 피사체가 프레임의 몇 %(선형)를 차지하게 할 것인가.
#                     학습 분포 실측(선형 중앙값 97~100%, p25 94~99%)에 맞춘 값이다.
MIN_AREA = 0.002    # 이보다 작은 덩어리는 먼지로 본다(전체 면적 대비)
MAX_LINEAR = 0.80   # 이미 이만큼 크면 자르지 않는다 — studio 를 건드려 망치지 않기 위한 안전장치
MAX_EDGE_TOUCH = 2  # 경계상자가 3변 이상에 닿으면 '배경을 잡았다'고 보고 포기


# ── 저수준 도구 ───────────────────────────────────────────────────────
def _box_mean(m: np.ndarray, k: int = 5) -> np.ndarray:
    """적분영상으로 k x k 이웃 평균 — scipy 없이 매끈하게."""
    p = np.pad(m.astype(np.float32), k // 2, mode="edge")
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = m.shape
    tot = c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]
    return tot / (k * k)


def _estimate_bg(a: np.ndarray) -> tuple[np.ndarray, int]:
    """네 모서리 패치 중앙값의 medoid 를 배경으로 본다. 동의한 모서리 수도 함께 준다.

    테두리 전체의 최빈색을 쓰면 피사체가 변에 걸칠 때 피사체를 배경으로 오인한다.
    """
    h, w = a.shape[:2]
    k = max(4, int(round(min(h, w) * CORNER)))
    patches = [a[:k, :k], a[:k, -k:], a[-k:, :k], a[-k:, -k:]]
    meds = np.stack([np.median(p.reshape(-1, 3), axis=0) for p in patches])
    dist = np.sqrt(((meds[:, None, :] - meds[None, :, :]) ** 2).sum(axis=2))
    agree = (dist <= 20.0).sum(axis=1)
    best = int(agree.argmax())
    return meds[dist[best] <= 20.0].mean(axis=0), int(agree[best])


def _otsu(v: np.ndarray, bins: int = 64) -> float:
    """[0,1] 값에 대한 Otsu 문턱 — 고정 문턱은 배경 밝기에 따라 무너진다."""
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = p.cumsum()
    m0 = (p * centers).cumsum()
    mt = m0[-1]
    denom = w0 * (1 - w0)
    with np.errstate(invalid="ignore", divide="ignore"):
        between = np.where(denom > 1e-12, (mt * w0 - m0) ** 2 / np.maximum(denom, 1e-12), 0.0)
    return float(centers[int(np.nanargmax(between))])


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int], max_iter: int = 400) -> np.ndarray:
    """seed 가 속한 연결요소만 남긴다(8-이웃 팽창 전파, scipy 없이).

    흩어진 점들의 전체 경계상자를 쓰면 배경 얼룩 하나가 상자를 화면 전체로 늘린다.
    """
    cur = np.zeros_like(mask, dtype=bool)
    cur[seed] = True
    for _ in range(max_iter):
        g = cur.copy()
        g[1:, :] |= cur[:-1, :]
        g[:-1, :] |= cur[1:, :]
        g[:, 1:] |= cur[:, :-1]
        g[:, :-1] |= cur[:, 1:]
        g[1:, 1:] |= cur[:-1, :-1]
        g[:-1, :-1] |= cur[1:, 1:]
        g[1:, :-1] |= cur[:-1, 1:]
        g[:-1, 1:] |= cur[1:, :-1]
        g &= mask
        if g.sum() == cur.sum():
            return g
        cur = g
    return cur


# ── saliency → 경계상자 ───────────────────────────────────────────────
def saliency(a: np.ndarray) -> np.ndarray:
    """색거리 + 엣지밀도, 중앙 가중. 입력은 작업해상도 RGB float 배열."""
    bg, _agree = _estimate_bg(a)
    color = np.sqrt(((a - bg) ** 2).sum(axis=2))

    g = a.mean(axis=2)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = np.abs(g[:, 2:] - g[:, :-2])
    gy[1:-1, :] = np.abs(g[2:, :] - g[:-2, :])
    edge = _box_mean(np.hypot(gx, gy), k=7)

    def norm(x: np.ndarray) -> np.ndarray:
        hi = float(np.percentile(x, 99))          # 최대값으로 나누면 점 하나가 전체를 눌러버린다
        return np.clip(x / hi, 0, 1) if hi > 1e-6 else np.zeros_like(x)

    sal = W_COLOR * norm(color) + W_EDGE * norm(edge)

    h, w = sal.shape
    yy = (np.arange(h) - (h - 1) / 2) / (h / 2)
    xx = (np.arange(w) - (w - 1) / 2) / (w / 2)
    r2 = yy[:, None] ** 2 + xx[None, :] ** 2
    prior = CENTER_FLOOR + (1 - CENTER_FLOOR) * np.exp(-r2 / (2 * CENTER_SIGMA ** 2))
    return _box_mean(sal * prior, k=5)


def subject_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    """원본 좌표계의 피사체 경계상자. 확신이 없으면 None — 자르지 않는 쪽이 안전하다."""
    W0, H0 = img.size
    small = img.convert("RGB").resize((WORK, WORK), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    sal = saliency(a)

    t = _otsu(sal)
    mask = sal > t
    mask &= _box_mean(mask, k=5) >= 0.4        # 점 노이즈 제거
    if not mask.any():
        return None

    seed = np.unravel_index(int((sal * mask).argmax()), sal.shape)
    comp = _component_from_seed(mask, (int(seed[0]), int(seed[1])))
    if comp.sum() < MIN_AREA * comp.size:
        return None

    ys, xs = np.where(comp.any(axis=1))[0], np.where(comp.any(axis=0))[0]
    y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1

    touch = (x0 == 0) + (y0 == 0) + (x1 == WORK) + (y1 == WORK)
    if touch >= MAX_EDGE_TOUCH + 1:
        return None
    if np.sqrt((x1 - x0) * (y1 - y0) / (WORK * WORK)) > MAX_LINEAR:
        return None                              # 이미 충분히 크다 → 손대지 않는다

    sx, sy = W0 / WORK, H0 / WORK
    return (int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy)))


def auto_crop(img: Image.Image, target_fill: float = TARGET_FILL) -> tuple[Image.Image, bool]:
    """피사체가 프레임의 target_fill(선형)을 차지하는 정사각 크롭. (이미지, 잘랐는가)."""
    box = subject_box(img)
    if box is None:
        return img, False
    W0, H0 = img.size
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) / max(target_fill, 1e-3)
    side = int(round(min(side, min(W0, H0))))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(round(min(max(cx - side / 2, 0), W0 - side)))
    top = int(round(min(max(cy - side / 2, 0), H0 - side)))
    return img.crop((left, top, left + side, top + side)), True


# ── 두 가지 배치 위치 ─────────────────────────────────────────────────
def prep_baseline(img: Image.Image) -> tuple[np.ndarray, bool]:
    """지금 그대로 — 224 로 늘리기만."""
    return E.to_tensor(img), False


def prep_client(img: Image.Image) -> tuple[np.ndarray, bool]:
    """브라우저가 원본 프레임에서 자른다. 잘라낸 뒤 224 로 줄이므로 화질 손실이 없다."""
    cropped, used = auto_crop(img)
    return E.to_tensor(cropped), used


def prep_server(img: Image.Image) -> tuple[np.ndarray, bool]:
    """서버는 이미 224 로 뭉개진 것만 받는다 — 그 안에서 잘라 다시 224 로 늘린다."""
    squashed = img.convert("RGB").resize((E.SIZE, E.SIZE), Image.BILINEAR)
    cropped, used = auto_crop(squashed)
    return E.to_tensor(cropped), used


PREPS = {"baseline": prep_baseline, "client": prep_client, "server": prep_server}


# ── 조건 ──────────────────────────────────────────────────────────────
GRAY = (150, 148, 143)   # eval_realworld.cond_far 가 쓰는 배경색


def compose(img: Image.Image, bg: tuple[int, int, int], occ: float) -> Image.Image:
    """_diag_potato.compose 와 같은 정의 — 피사체가 프레임의 occ(선형)를 차지."""
    if occ >= 0.999:
        return img
    w, h = img.size
    canvas = Image.new("RGB", (round(w / occ), round(h / occ)), bg)
    canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


def clutter_bg(w: int, h: int, seed: int = 3) -> Image.Image:
    """복잡한 배경 — 흙·풀·상자 널판이 섞인 마당. 평평한 색이 아니라서 크롭기를 흔든다."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (max(h // 16, 4), max(w // 16, 4), 3), dtype=np.uint8)
    im = Image.fromarray(base).resize((w, h), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    for _ in range(24):                       # 잡동사니 얼룩
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = int(rng.integers(w // 20, w // 6))
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  fill=tuple(int(v) for v in rng.integers(30, 220, 3)))
    for _ in range(10):                       # 상자 널판 모서리(직선 엣지)
        y = int(rng.integers(0, h))
        d.line([0, y, w, int(rng.integers(0, h))], width=int(rng.integers(2, 7)),
               fill=tuple(int(v) for v in rng.integers(40, 200, 3)))
    a = np.asarray(im, np.int16) + rng.integers(-25, 25, (h, w, 3), dtype=np.int16)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def cond_clutter(img: Image.Image, occ: float) -> Image.Image:
    """복잡한 배경 위에 occ 크기로 올린다."""
    w, h = img.size
    if occ >= 0.999:
        return img
    cw, ch = round(w / occ), round(h / occ)
    bg = clutter_bg(cw, ch)
    bg.paste(img, ((cw - w) // 2, (ch - h) // 2))
    return bg


def subject_alpha(img: Image.Image) -> Image.Image:
    """스튜디오 사진에서 피사체만 남기는 알파 — 흰 배경과의 색거리로 자른다."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    bg, _ = _estimate_bg(a)
    m = np.sqrt(((a - bg) ** 2).sum(axis=2)) > 22.0
    m &= _box_mean(m, k=5) >= 0.5
    return Image.fromarray((m * 255).astype(np.uint8), mode="L")


def cut_compose(img: Image.Image, bg: object, occ: float) -> Image.Image:
    """★정직한 원거리 모사 — 피사체만 오려서 배경에 얹는다(흰 테두리 없음).

    compose() 는 '흰 배경 타일'을 통째로 붙이므로, 크롭기가 과일이 아니라 타일 경계를
    찾아도 100% 가 나온다. 그 착시를 걷어내고 재기 위한 조건이다.
    """
    w, h = img.size
    cw, ch = round(w / occ), round(h / occ)
    canvas = clutter_bg(cw, ch) if bg == "clutter" else Image.new("RGB", (cw, ch), bg)  # type: ignore[arg-type]
    canvas.paste(img, ((cw - w) // 2, (ch - h) // 2), subject_alpha(img))
    return canvas


CONDS: dict[str, tuple[str, object]] = {
    "studio": ("학습과 같은 조건", lambda im: im),
    "far": ("멀리서 촬영(회색)", E.cond_far),
    "fill50": ("차지 50%(회색)", lambda im: compose(im, GRAY, 0.50)),
    "fill35": ("차지 35%(회색)", lambda im: compose(im, GRAY, 0.35)),
    "fill25": ("차지 25%(회색)", lambda im: compose(im, GRAY, 0.25)),
    "fill15": ("차지 15%(회색)", lambda im: compose(im, GRAY, 0.15)),
    "background": ("흙 배경(꽉 참)", E.cond_background),
    "clutter100": ("복잡배경·꽉 참", lambda im: cond_clutter(im, 1.0)),
    "clutter50": ("복잡배경 50%", lambda im: cond_clutter(im, 0.50)),
    "clutter25": ("복잡배경 25%", lambda im: cond_clutter(im, 0.25)),
    # 오려붙인 조건 — 흰 테두리 착시를 걷어낸 값. 위 조건들보다 이쪽이 현장에 가깝다.
    "cut50": ("오림 50%(회색)", lambda im: cut_compose(im, GRAY, 0.50)),
    "cut35": ("오림 35%(회색)", lambda im: cut_compose(im, GRAY, 0.35)),
    "cut25": ("오림 25%(회색)", lambda im: cut_compose(im, GRAY, 0.25)),
    "cut15": ("오림 15%(회색)", lambda im: cut_compose(im, GRAY, 0.15)),
    "cutsoil25": ("오림 25%(흙)", lambda im: cut_compose(im, (122, 104, 78), 0.25)),
    "cutclutter25": ("오림 25%(복잡)", lambda im: cut_compose(im, "clutter", 0.25)),
}


# ── 측정 ──────────────────────────────────────────────────────────────
def logits(tensors: list[np.ndarray]) -> np.ndarray:
    x = np.concatenate(tensors, axis=0)
    out = SESSION.run(None, {SESSION.get_inputs()[0].name: x})
    names = [o.name for o in SESSION.get_outputs()]
    return np.asarray(out[names.index("item_logits")])


def topk(lg: np.ndarray, truth: str, k: int) -> float:
    order = np.argsort(-lg, axis=1)[:, :k]
    return float((order == ITEMS.index(truth)).any(axis=1).mean())


# ── 검증: 크롭기가 아는 답을 맞히는가 ─────────────────────────────────
def _disc(size: int, occ: float, at: tuple[float, float] = (0.5, 0.5),
          bg: tuple[int, int, int] = GRAY, fg: tuple[int, int, int] = (196, 180, 136)) -> Image.Image:
    im = Image.new("RGB", (size, size), bg)
    r = size * occ / 2
    cx, cy = size * at[0], size * at[1]
    ImageDraw.Draw(im).ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    return im


def _fill_of(img: Image.Image) -> float:
    """이미지 안에서 피사체가 차지하는 선형 비율 — 크롭 결과를 독립적으로 잰다."""
    a = np.asarray(img.convert("RGB").resize((128, 128), Image.BILINEAR), np.float32)
    bg, _ = _estimate_bg(a)
    m = np.sqrt(((a - bg) ** 2).sum(axis=2)) > 20.0
    m &= _box_mean(m, 5) >= 0.5
    if m.sum() < 0.001 * m.size:
        return 0.0
    ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return float(np.sqrt((xs[-1] - xs[0] + 1) * (ys[-1] - ys[0] + 1)) / 128)


def _self_test(verbose: bool = True) -> int:
    checks = 0

    # 1) 아는 크기·위치의 원반을 찾아내는가
    for occ in (0.50, 0.25, 0.15):
        box = subject_box(_disc(512, occ))
        assert box, f"차지 {occ}: 피사체를 못 찾았다"
        side = max(box[2] - box[0], box[3] - box[1]) / 512
        assert abs(side - occ) < 0.06, f"차지 {occ}: 상자가 {side:.2f} 로 어긋난다 {box}"
        checks += 1

    # 2) 중앙이 아닌 곳도 찾는가 (중앙 가중이 피사체를 놓치게 만들면 안 된다)
    box = subject_box(_disc(512, 0.20, at=(0.28, 0.72)))
    assert box, "치우친 피사체를 못 찾았다"
    cx, cy = (box[0] + box[2]) / 2 / 512, (box[1] + box[3]) / 2 / 512
    assert abs(cx - 0.28) < 0.06 and abs(cy - 0.72) < 0.06, f"치우친 피사체 중심 오차 {cx:.2f},{cy:.2f}"
    checks += 1

    # 3) 자른 뒤 실제로 TARGET_FILL 만큼 차게 되는가 (독립 측정기로)
    cropped, used = auto_crop(_disc(512, 0.25))
    assert used, "잘라야 하는데 안 잘랐다"
    got = _fill_of(cropped)
    assert abs(got - TARGET_FILL) < 0.10, f"크롭 후 차지비율 {got:.2f} (목표 {TARGET_FILL})"
    checks += 1

    # 4) 음성 대조군 — 빈 화면에서 무언가를 잘라내면 안 된다
    _, used = auto_crop(Image.new("RGB", (512, 512), GRAY))
    assert not used, "빈 화면에서 피사체를 찾아 잘랐다"
    checks += 1

    # 5) 안전장치 — 이미 꽉 찬 실제 학습 이미지는 건드리지 않는다
    picks = E.sample_images(3)
    untouched = 0
    for item, files in picks.items():
        for f in files:
            im = Image.open(f)
            out, used = auto_crop(im)
            untouched += (not used)
            assert not used or _fill_of(out) > 0.80, f"{item}: 꽉 찬 사진을 {_fill_of(out):.2f} 로 잘랐다"
    assert untouched >= 12, f"꽉 찬 사진 15장 중 {untouched}장만 건드리지 않았다(안전장치가 헐겁다)"
    checks += 1

    # 6) 정답 상자와의 IoU — 합성이므로 피사체가 어디 붙었는지 정확히 안다.
    #    (픽셀 MAE 로 재면 '의도한 5% 여백'까지 오차로 세어 헛돈다. 위치·크기를 직접 본다.)
    def _iou(a: tuple, b: tuple) -> float:
        ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / max(union, 1)

    ious, closer = [], []
    for item, files in picks.items():
        orig = Image.open(files[0]).convert("RGB")
        far = compose(orig, GRAY, 0.25)
        w = far.size[0]
        off = (w - orig.size[0]) // 2
        truth = (off, off, off + orig.size[0], off + orig.size[1])
        box = subject_box(far)
        assert box, f"{item}: far 에서 피사체를 못 찾았다"
        ious.append(_iou(box, truth))
        back, used = auto_crop(far)
        assert used, f"{item}: far 를 자르지 않았다"
        a = np.asarray(orig.resize((64, 64)), np.float32)
        d_crop = float(np.abs(a - np.asarray(back.convert("RGB").resize((64, 64)), np.float32)).mean())
        d_far = float(np.abs(a - np.asarray(far.resize((64, 64)), np.float32)).mean())
        # 방향만 본다 — 감자처럼 배경색과 비슷한 피사체는 픽셀거리가 원래 잘 안 벌어진다.
        closer.append(d_crop < d_far * 0.9)
    assert min(ious) > 0.65, f"정답 상자와 어긋난다: IoU 최소 {min(ious):.2f}"
    assert all(closer), "잘라낸 그림이 원본에 더 가까워지지 않았다"
    checks += 1

    if verbose:
        print(f"✅ 검증 {checks}건 통과 — 아는 크기(±6%p)·치우친 피사체·목표 차지비율·"
              f"빈 화면 무동작·꽉 찬 사진 무동작·far 정답상자 IoU {np.mean(ious):.2f}")
    return checks


def _fail_test() -> None:
    """검사가 헛돌지 않는지 — 크롭기를 일부러 망가뜨리면 반드시 걸려야 한다."""
    global MAX_LINEAR
    keep = MAX_LINEAR
    try:
        MAX_LINEAR = 0.0        # 어떤 상자도 통과 못 하게 → 항상 원본 반환
        try:
            _self_test(verbose=False)
        except AssertionError:
            print("✅ 역검증 — 안전장치를 망가뜨리자 검사가 실패로 바뀐다(검사가 살아 있다)")
            return
        raise SystemExit("❌ 역검증 실패 — 크롭기를 망가뜨렸는데 검사가 통과했다")
    finally:
        MAX_LINEAR = keep


# ── 본 측정 ───────────────────────────────────────────────────────────
def measure(n: int, cond_keys: list[str], preps: list[str]) -> dict:
    picks = E.sample_images(n)          # baseline 과 같은 표본·같은 seed(7)
    cache = {it: [Image.open(f).convert("RGB") for f in fs] for it, fs in picks.items()}
    out: dict = {}
    for key in cond_keys:
        label, fn = CONDS[key]
        out[key] = {"label": label, "preps": {}}
        staged = {it: [fn(im) for im in ims] for it, ims in cache.items()}
        for pname in preps:
            prep = PREPS[pname]
            per_item, used_n, used_d = {}, 0, 0
            for item, imgs in staged.items():
                pairs = [prep(im) for im in imgs]
                lg = logits([t for t, _ in pairs])
                used_n += sum(u for _, u in pairs)
                used_d += len(pairs)
                per_item[item] = {"top1": topk(lg, item, 1), "top3": topk(lg, item, 3)}
            out[key]["preps"][pname] = {
                "per_item": per_item,
                "top1": float(np.mean([v["top1"] for v in per_item.values()])),
                "top3": float(np.mean([v["top3"] for v in per_item.values()])),
                "crop_rate": used_n / max(used_d, 1),
            }
    return out


def report(res: dict, preps: list[str]) -> None:
    print("\n" + "=" * 100)
    print("조건별 전체 top-1 / top-3   (품목 5종 평균, 같은 표본)")
    print("=" * 100)
    head = f"{'조건':<16}" + "".join(f"{p:>20}" for p in preps) + f"{'크롭적용률(client)':>18}"
    print(head)
    print("-" * len(head))
    for key, r in res.items():
        cells = ""
        for p in preps:
            v = r["preps"][p]
            cells += f"{v['top1']:>9.0%} /{v['top3']:>8.0%}"
        cr = r["preps"].get("client", {}).get("crop_rate", 0)
        print(f"{r['label']:<16}" + cells + f"{cr:>17.0%}")

    for p in preps:
        if p == "baseline":
            continue
        print(f"\n{'=' * 100}\n품목별 — baseline → {p}   (top-1 / top-3)\n{'=' * 100}")
        head = f"{'조건':<16}" + "".join(f"{it:>22}" for it in ITEMS)
        print(head)
        print("-" * len(head))
        for key, r in res.items():
            row = f"{r['label']:<16}"
            for it in ITEMS:
                b = r["preps"]["baseline"]["per_item"][it]
                v = r["preps"][p]["per_item"][it]
                row += f"  {b['top1']:>3.0%}/{b['top3']:<3.0%}→{v['top1']:>3.0%}/{v['top3']:<3.0%}"
            print(row)

    print(f"\n{'=' * 100}\n감자 far — 지금 0% 인 지점\n{'=' * 100}")
    if "far" in res:
        for p in preps:
            v = res["far"]["preps"][p]["per_item"]["감자"]
            print(f"  {p:<10} top-1 {v['top1']:>5.0%}   top-3 {v['top3']:>5.0%}")


def ablation(n: int) -> None:
    """여백(TARGET_FILL)과 구성요소의 민감도 — 한 값에 요행으로 기대고 있지 않은지."""
    global TARGET_FILL, W_COLOR, W_EDGE
    picks = E.sample_images(n)
    cache = {it: [Image.open(f).convert("RGB") for f in fs] for it, fs in picks.items()}
    keys = ["studio", "cut25", "cut15", "clutter25"]   # 오림 조건이 현장에 더 가깝다
    keep_fill, keep_c, keep_e = TARGET_FILL, W_COLOR, W_EDGE

    print("\n" + "=" * 100)
    print("민감도 — 여백(TARGET_FILL) 과 saliency 구성요소   (전체 top-1/top-3)")
    print("=" * 100)
    head = f"{'설정':<22}" + "".join(f"{CONDS[k][0]:>20}" for k in keys)
    print(head)
    print("-" * len(head))
    try:
        settings = [(f"fill={v}", v, keep_c, keep_e) for v in (1.00, 0.95, 0.85, 0.70)]
        settings += [("색만(엣지 0)", keep_fill, 1.0, 0.0), ("엣지만(색 0)", keep_fill, 0.0, 1.0)]
        for name, tf, wc, we in settings:
            TARGET_FILL, W_COLOR, W_EDGE = tf, wc, we
            row = f"{name:<22}"
            for k in keys:
                fn = CONDS[k][1]
                t1, t3 = [], []
                for item, imgs in cache.items():
                    lg = logits([prep_client(fn(im))[0] for im in imgs])
                    t1.append(topk(lg, item, 1))
                    t3.append(topk(lg, item, 3))
                row += f"{np.mean(t1):>9.0%} /{np.mean(t3):>8.0%}"
            print(row)
    finally:
        TARGET_FILL, W_COLOR, W_EDGE = keep_fill, keep_c, keep_e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="품목당 표본 수(baseline 과 같은 30)")
    ap.add_argument("--conditions", default="")
    ap.add_argument("--preps", default="baseline,client,server")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        _fail_test()
        return 0

    _self_test()
    _fail_test()

    if args.ablation:
        ablation(args.n)
        return 0

    keys = [k.strip() for k in args.conditions.split(",") if k.strip()] or list(CONDS)
    preps = [p.strip() for p in args.preps.split(",") if p.strip()]
    print(f"\n검증 분할 · 품목당 {args.n}장 · E.sample_images(seed=7) — baseline 과 같은 표본")
    res = measure(args.n, keys, preps)
    report(res, preps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
