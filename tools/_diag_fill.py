"""학습 데이터에서 피사체가 프레임을 얼마나 차지하는가 — 분포 측정(scratch).

왜 재는가: '멀리서 촬영'하면 top-1 이 급락한다는 것은 이미 측정됐다.
급락은 차지비율(_diag_potato.compose 의 occ, 선형 비율) 0.75~0.50 사이에서 시작한다.
그런데 occ=1.0 은 '학습 이미지가 프레임을 꽉 채운 상태'이지 '피사체가 꽉 찬 상태'가 아니다.
학습 이미지 안에서 피사체가 이미 몇 %만 차지하고 있는지 알아야
"폰으로 찍을 때 피사체가 화면의 몇 %를 채워야 하는가"라는 절대 기준이 나온다.

    python tools/_diag_fill.py --self-test          # 측정기를 먼저 검증
    python tools/_diag_fill.py --n 100              # 본 측정 (train/valid × 5품목)
    python tools/_diag_fill.py --overlay OUT.png    # 경계상자를 그린 눈검사 시트

전제: AI-Hub 149 는 흰(또는 밝은 회색) 배경 스튜디오 촬영이고,
tools/aihub_ingest.py 는 원본 1000x1000 을 자르지 않고 224 로 줄이기만 한다.
따라서 여기서 재는 비율은 원본 촬영 구도의 비율과 같다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_realworld as E  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = E.ROOT
ITEMS = E.ITEMS
TRAIN = ROOT / "data" / "onfarm_cv" / "train"
VALID = ROOT / "data" / "onfarm_cv" / "valid"

CORNER = 0.10   # 배경색을 추정할 네 모서리 패치 크기(변 길이 대비)
THRESH = 20.0   # 배경색과의 RGB 거리가 이보다 크면 피사체
MIN_NEIGH = 0.5  # 5x5 이웃의 절반 이상이 피사체여야 살아남는다(점 노이즈 제거)


# ── 배경 추정 ─────────────────────────────────────────────────────────
def estimate_bg(a: np.ndarray) -> tuple[np.ndarray, int]:
    """네 모서리 패치의 중앙값 중 medoid = 배경. 함께 '몇 개 모서리가 동의하는지'를 돌려준다.

    테두리 전체의 최빈색을 쓰면 피사체가 변에 잘려 테두리를 뒤덮을 때 피사체 색을 배경으로
    잘못 잡는다. 둥근 피사체는 네 모서리를 동시에 덮기 어려우므로 모서리를 쓴다.
    동의하는 모서리가 2개 미만이면 배경을 못 찾은 것으로 보고 측정 실패로 처리한다.
    """
    h, w = a.shape[:2]
    k = max(4, int(round(min(h, w) * CORNER)))
    patches = [a[:k, :k], a[:k, -k:], a[-k:, :k], a[-k:, -k:]]
    meds = np.stack([np.median(p.reshape(-1, 3), axis=0) for p in patches])
    dist = np.sqrt(((meds[:, None, :] - meds[None, :, :]) ** 2).sum(axis=2))
    agree = (dist <= THRESH).sum(axis=1)          # 자기 자신 포함
    best = int(agree.argmax())
    members = meds[dist[best] <= THRESH]
    return members.mean(axis=0), int(agree[best])


# ── 피사체 마스크 ─────────────────────────────────────────────────────
def _box_mean(m: np.ndarray, k: int = 5) -> np.ndarray:
    """적분영상으로 k x k 이웃 평균 — scipy 없이 점 노이즈를 지우기 위해."""
    p = np.pad(m.astype(np.float32), k // 2, mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = m.shape
    tot = c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]
    return tot / (k * k)


def foreground(a: np.ndarray, thresh: float = THRESH) -> tuple[np.ndarray, int]:
    bg, agree = estimate_bg(a)
    d = np.sqrt(((a - bg) ** 2).sum(axis=2))
    raw = d > thresh
    return raw & (_box_mean(raw) >= MIN_NEIGH), agree


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """피사체 경계상자(x0, y0, x1, y1). 피사체가 없으면 None — 조용히 100% 로 세지 않게."""
    if mask.sum() < 0.001 * mask.size:
        return None
    ys, xs = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def measure(img: Image.Image, thresh: float = THRESH) -> dict[str, float] | None:
    """경계상자가 이미지 면적의 몇 %인가, 그리고 그에 해당하는 선형 비율.

    배경을 못 찾았거나(모서리 동의 2개 미만) 피사체가 안 잡히면 None — 실패는 실패로 센다.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = a.shape[:2]
    m, agree = foreground(a, thresh)
    if agree < 2:
        return None
    box = bbox(m)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    area = (x1 - x0) * (y1 - y0) / (w * h)
    return {
        "bbox_area_pct": 100 * area,
        "linear_pct": 100 * float(np.sqrt(area)),   # occ(선형 비율)과 같은 축
        "bbox_w_pct": 100 * (x1 - x0) / w,
        "bbox_h_pct": 100 * (y1 - y0) / h,
        "mask_area_pct": 100 * float(m.sum()) / (w * h),
        "touch_edge": float(x0 == 0 or y0 == 0 or x1 == w or y1 == h),
    }


# ── 측정기 검증 ───────────────────────────────────────────────────────
def _synth(bg: tuple[int, int, int], fg: tuple[int, int, int], side: int,
           at: tuple[int, int] | None = None, size: int = 224) -> Image.Image:
    im = Image.new("RGB", (size, size), bg)
    xy = at or ((size - side) // 2, (size - side) // 2)
    ImageDraw.Draw(im).rectangle([xy[0], xy[1], xy[0] + side - 1, xy[1] + side - 1], fill=fg)
    return im


def _self_test() -> None:
    """측정기가 아는 답을 맞히는지. 하나라도 틀리면 본 측정은 의미가 없다."""
    checks = 0

    # 1) 아는 크기의 사각형 — 흰 배경
    for side, want in [(200, 89.3), (168, 75.0), (112, 50.0), (56, 25.0)]:
        got = measure(_synth((255, 255, 255), (200, 60, 40), side))
        assert got and abs(got["linear_pct"] - want) < 1.5, f"흰배경 {side}px: {got}"
        checks += 1

    # 2) 회색 배경 + 밝은 피사체(흰 양파처럼 대비가 약한 경우)
    got = measure(_synth((200, 205, 210), (240, 240, 238), 112))
    assert got and abs(got["linear_pct"] - 50.0) < 1.5, f"저대비: {got}"
    checks += 1

    # 3) 변에 잘린 피사체 — 테두리 최빈색이 배경을 계속 잡아내는가
    got = measure(_synth((255, 255, 255), (120, 90, 40), 200, at=(-40, 10)))
    assert got and abs(got["bbox_w_pct"] - 160 / 224 * 100) < 1.5, f"잘린 피사체: {got}"
    checks += 1

    # 4) 음성 대조군 — 빈 배경은 '피사체 없음'이어야 한다(0%가 아니라 None)
    assert measure(Image.new("RGB", (224, 224), (255, 255, 255))) is None, "빈 이미지에서 피사체를 찾았다"
    checks += 1

    # 4b) 피사체가 프레임을 완전히 덮은 경우 — 작은 값을 조용히 내놓으면 안 된다.
    #     (배경이 안 보이므로 '실패' 또는 '거의 100%' 둘 중 하나여야 한다)
    real0 = next(p for g in sorted((TRAIN / ITEMS[0]).iterdir()) if g.is_dir() for p in g.glob("*.jpg"))
    full = Image.open(real0).convert("RGB").resize((448, 448)).crop((112, 112, 336, 336))
    got = measure(full)
    assert got is None or got["linear_pct"] > 90, f"꽉 찬 피사체를 {got} 로 축소보고했다"
    checks += 1

    # 5) 실측치와 occ 축의 연결 — 실제 학습 이미지를 occ=0.5 로 축소하면
    #    경계상자 선형 비율도 정확히 절반이 되어야 한다.
    im = Image.open(real0)
    base = measure(im)
    canvas = Image.new("RGB", (448, 448), (255, 255, 255))  # 흰 배경이라야 피사체만 전경이 된다
    canvas.paste(im, (112, 112))
    half = measure(canvas)
    assert base and half, "실이미지 측정 실패"
    assert abs(half["linear_pct"] - base["linear_pct"] * 0.5) < 2.0, \
        f"occ 축과 어긋난다: {base['linear_pct']:.1f} → {half['linear_pct']:.1f}"
    checks += 1

    print(f"✅ 검증 {checks}건 통과 — 아는 크기(±1.5%p), 저대비, 잘린 피사체, 빈 이미지, occ 축 연결")


def _fail_test() -> None:
    """검사가 헛돌지 않는지 — 일부러 망가뜨린 측정기는 반드시 걸려야 한다."""
    global THRESH
    keep = THRESH
    try:
        THRESH = 500.0  # 어떤 픽셀도 피사체가 아니게 만든다
        got = measure(_synth((255, 255, 255), (200, 60, 40), 112), thresh=THRESH)
        assert got is None, f"망가뜨렸는데도 값이 나왔다: {got}"
        print("✅ 역검증 — 문턱을 망가뜨리자 측정이 실패로 바뀐다(검사가 살아 있다)")
    finally:
        THRESH = keep


# ── 표본 ──────────────────────────────────────────────────────────────
def sample(root: Path, n: int, seed: int = 11) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    out: dict[str, list[Path]] = {}
    for item in ITEMS:
        base = root / item
        if not base.exists():
            continue
        files = [p for g in sorted(base.iterdir()) if g.is_dir() for p in g.glob("*.jpg")]
        rng.shuffle(files)
        out[item] = files[:n]
    return out


def summarize(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    p = np.percentile(a, [5, 25, 50, 75, 95])
    return {"n": len(a), "p5": p[0], "p25": p[1], "median": p[2], "p75": p[3], "p95": p[4],
            "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean())}


def overlay_sheet(out: Path, per_row: int = 8, seed: int = 11) -> None:
    """경계상자를 그려 눈으로 확인 — 숫자가 그럴듯해 보여도 마스크가 틀렸을 수 있다."""
    S = 168
    sheet = Image.new("RGB", (per_row * S, len(ITEMS) * S), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    picks = sample(TRAIN, per_row, seed=seed)
    for r, item in enumerate(ITEMS):
        for c, f in enumerate(picks.get(item, [])):
            im = Image.open(f).convert("RGB")
            got = measure(im)
            sheet.paste(im.resize((S, S)), (c * S, r * S))
            if got:
                a = np.asarray(im, dtype=np.float32)
                x0, y0, x1, y1 = bbox(foreground(a)[0])  # type: ignore[misc]
                k = S / im.size[0]
                draw.rectangle([c * S + x0 * k, r * S + y0 * k, c * S + x1 * k, r * S + y1 * k],
                               outline=(255, 0, 255), width=2)
                draw.text((c * S + 4, r * S + 4), f"{got['bbox_area_pct']:.0f}%", fill=(255, 0, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"눈검사 시트: {out}")


# ── 본 측정 ───────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="분할·품목당 표본 수")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--overlay", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        _fail_test()
        return 0
    if args.overlay:
        overlay_sheet(args.overlay)
        return 0

    _self_test()
    _fail_test()

    results: dict[str, dict[str, dict[str, float]]] = {}
    raw: dict[str, dict[str, list[float]]] = {}
    for split, root in [("train", TRAIN), ("valid", VALID)]:
        picks = sample(root, args.n)
        results[split] = {}
        raw[split] = {}
        for item, files in picks.items():
            got = [measure(Image.open(f)) for f in files]
            ok = [g for g in got if g]
            if not ok:
                continue
            raw[split][item] = [g["linear_pct"] for g in ok]
            s = summarize([g["bbox_area_pct"] for g in ok])
            s["linear_median"] = float(np.median([g["linear_pct"] for g in ok]))
            s["linear_p5"] = float(np.percentile([g["linear_pct"] for g in ok], 5))
            s["mask_median"] = float(np.median([g["mask_area_pct"] for g in ok]))
            s["touch_edge_pct"] = 100 * float(np.mean([g["touch_edge"] for g in ok]))
            s["failed"] = len(got) - len(ok)
            results[split][item] = s

    print(f"\n학습·검증 데이터에서 피사체가 프레임을 차지하는 비율 · 분할·품목당 {args.n}장\n")
    hdr = f"{'분할':<6}{'품목':<6}{'면적%중앙':>10}{'p5':>7}{'p25':>7}{'p75':>7}{'p95':>7}{'선형%중앙':>10}{'마스크%':>8}{'변접촉':>7}"
    print(hdr)
    print("-" * len(hdr))
    for split in results:
        for item, s in results[split].items():
            print(f"{split:<6}{item:<6}{s['median']:>10.1f}{s['p5']:>7.1f}{s['p25']:>7.1f}"
                  f"{s['p75']:>7.1f}{s['p95']:>7.1f}{s['linear_median']:>10.1f}"
                  f"{s['mask_median']:>8.1f}{s['touch_edge_pct']:>6.0f}%")

    # 전체 종합
    all_lin = [v for sp in raw.values() for vs in sp.values() for v in vs]
    lin_med = float(np.median(all_lin))
    lin_p5 = float(np.percentile(all_lin, 5))
    print(f"\n전체 {len(all_lin)}장 선형 차지비율: 중앙값 {lin_med:.1f}%  "
          f"p5 {lin_p5:.1f}%  p25 {np.percentile(all_lin, 25):.1f}%  p95 {np.percentile(all_lin, 95):.1f}%")

    # 학습분포의 꼬리 — 급락 구간에 학습 예시가 실제로 있었는가
    arr = np.asarray(all_lin)
    print("\n학습·검증 분포에서 선형 차지비율이 X 미만인 장수 (모델이 본 적 있는가)")
    for t in [90, 85, 80, 75, 70, 60, 50, 35]:
        print(f"  <{t}% :  {100 * (arr < t).mean():>5.1f}%  ({int((arr < t).sum())}장)")

    # 급락 지점과의 대조 — occ 는 '학습 이미지'의 축소율이므로 실제 피사체 비율로 환산한다
    print("\n급락 지점 환산 (occ = 학습 이미지 축소율, 실제 피사체 선형비율 = 중앙 학습비율 × occ)")
    hdr2 = f"{'occ':>6}{'top-1':>8}{'top-3':>8}{'피사체 화면폭%':>15}{'프레임 면적%':>14}{'학습분포 중 이보다 작은 것':>26}"
    print(hdr2)
    for occ, t1, t3 in [(1.00, 100, 100), (0.75, 100, 100), (0.50, 75, 100),
                        (0.35, 55, 87), (0.25, 57, 80), (0.15, 43, 73)]:
        eff = lin_med * occ
        print(f"{occ:>6.2f}{t1:>7}%{t3:>7}%{eff:>14.0f}%{eff * eff / 100:>13.0f}%"
              f"{100 * (arr < eff).mean():>24.1f}%")

    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
