"""감자가 멀어지면 왜 '사과'가 되는가 — 원인 분해(scratch).

핵심 질문: '배경 때문'인가 '피사체 크기 때문'인가.
둘을 교차시켜(배경 5색 × 차지비율 6단계) 각각의 기여를 숫자로 가른다.

    python tools/_diag_potato.py                # 전부
    python tools/_diag_potato.py --only grid    # 일부만
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_realworld as E  # noqa: E402

import onnxruntime as ort  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ITEMS = E.ITEMS
SESSION = ort.InferenceSession(str(E.MODEL), providers=["CPUExecutionProvider"])

BACKGROUNDS = {
    "회색(150,148,143)": (150, 148, 143),   # eval_realworld 의 far 가 쓰는 색
    "흰색(245,245,245)": (245, 245, 245),
    "검정(20,20,20)": (20, 20, 20),
    "갈색흙(122,104,78)": (122, 104, 78),
    "초록(70,110,60)": (70, 110, 60),
}
OCCUPANCIES = [1.00, 0.75, 0.50, 0.35, 0.25, 0.15]


# ── 도구 ──────────────────────────────────────────────────────────────
def logits(imgs: list[Image.Image]) -> np.ndarray:
    """item_logits 를 배치로 얻는다."""
    x = np.concatenate([E.to_tensor(im) for im in imgs], axis=0)
    out = SESSION.run(None, {SESSION.get_inputs()[0].name: x})
    names = [o.name for o in SESSION.get_outputs()]
    return np.asarray(out[names.index("item_logits")])


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def compose(img: Image.Image, bg: tuple[int, int, int], occ: float) -> Image.Image:
    """피사체가 프레임의 occ(선형 비율)만 차지하도록 배경 위에 올린다.

    occ=1.0 이면 원본 그대로(=studio). occ=0.25 면 eval_realworld 의 far 와 같은 크기.
    """
    if occ >= 0.999:
        return img
    w, h = img.size
    canvas = Image.new("RGB", (round(w / occ), round(h / occ)), bg)
    canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


def blank(bg: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (224, 224), bg)


def detail_only(img: Image.Image, k: int) -> Image.Image:
    """크기는 그대로 프레임을 꽉 채우되 해상도(디테일)만 k×k 로 죽인다.

    '작아지면 픽셀이 없어서 못 맞히는 것'과 '작아지면 배경에 파묻혀 못 맞히는 것'을 가른다.
    """
    if k >= 224:
        return img
    return img.resize((k, k), Image.LANCZOS).resize((224, 224), Image.BILINEAR)


def stats(img: Image.Image) -> dict[str, float]:
    """224x224 텐서가 되기 직전의 간단한 화상 지표."""
    a = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32)
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    g = a.mean(axis=2)
    edge = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2
    return {
        "R": float(a[..., 0].mean()), "G": float(a[..., 1].mean()), "B": float(a[..., 2].mean()),
        "밝기": float(g.mean()), "채도": float(sat.mean() * 100),
        "R-G": float(a[..., 0].mean() - a[..., 1].mean()),
        "엣지": edge, "표준편차": float(g.std()),
    }


def topk_rate(lg: np.ndarray, truth: str, k: int) -> float:
    order = np.argsort(-lg, axis=1)[:, :k]
    ti = ITEMS.index(truth)
    return float((order == ti).any(axis=1).mean())


def load(files: list[Path]) -> list[Image.Image]:
    return [Image.open(f).convert("RGB") for f in files]


# ── 1. 감자 far 의 logit 분포와, 확대했을 때의 회복 ───────────────────
def test_logits(picks: dict[str, list[Path]]) -> None:
    print("\n" + "=" * 78)
    print("1. 감자 far 의 item_logits — 그리고 같은 이미지를 중앙 확대하면")
    print("=" * 78)

    files = picks["감자"]
    imgs = load(files)
    far = [E.cond_far(im) for im in imgs]
    # far 로 만든 사진에서, 피사체가 있는 중앙 25% 영역만 다시 잘라낸다(=디지털 확대)
    zoom = []
    for f in far:
        w, h = f.size
        s = w // 4
        zoom.append(f.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)))

    for name, batch in (("studio(원본)", imgs), ("far(멀리)", far), ("far→중앙확대", zoom)):
        lg = logits(batch)
        pr = softmax(lg)
        print(f"\n[{name}]  n={len(batch)}")
        print(f"  평균 logit : " + "  ".join(f"{it}={lg[:, i].mean():+6.2f}" for i, it in enumerate(ITEMS)))
        print(f"  평균 확률  : " + "  ".join(f"{it}={pr[:, i].mean():5.1%}" for i, it in enumerate(ITEMS)))
        pred = [ITEMS[i] for i in lg.argmax(1)]
        cnt = {it: pred.count(it) for it in ITEMS if pred.count(it)}
        print(f"  예측 분포  : {cnt}")
        print(f"  감자 top-1={topk_rate(lg, '감자', 1):.0%}  top-3={topk_rate(lg, '감자', 3):.0%}"
              f"  |  감자 순위 평균 {np.mean([list(np.argsort(-r)).index(ITEMS.index('감자')) + 1 for r in lg]):.2f}위")
        print(f"  감자-사과 logit 차 평균 {(lg[:, ITEMS.index('감자')] - lg[:, ITEMS.index('사과')]).mean():+.2f}")

    print("\n  ※ far 에서 감자 logit 이 어디로 갔는지: 사과 logit 은 유지되고 감자만 무너지는지 확인")


# ── 2. 감자 vs 사과 텐서 통계 ─────────────────────────────────────────
def test_stats(picks: dict[str, list[Path]]) -> None:
    print("\n" + "=" * 78)
    print("2. 224x224 화상 통계 — 감자/사과, studio vs far, 그리고 배경 단독")
    print("=" * 78)
    rows: list[tuple[str, dict[str, float]]] = []
    for item in ("감자", "사과", "배", "감귤", "양파"):
        imgs = load(picks[item])
        for tag, batch in ((f"{item} studio", imgs), (f"{item} far", [E.cond_far(i) for i in imgs])):
            ss = [stats(i) for i in batch]
            rows.append((tag, {k: float(np.mean([s[k] for s in ss])) for k in ss[0]}))
    for label, rgb in BACKGROUNDS.items():
        rows.append((f"배경만 {label}", stats(blank(rgb))))

    keys = ["R", "G", "B", "밝기", "채도", "R-G", "엣지", "표준편차"]
    w = max(len(r[0]) for r in rows) + 2
    print(f"\n{'':<{w}}" + "".join(f"{k:>9}" for k in keys))
    for label, s in rows:
        print(f"{label:<{w}}" + "".join(f"{s[k]:>9.1f}" for k in keys))

    def vec(tag: str) -> np.ndarray:
        d = dict(rows)[tag]
        return np.array([d[k] for k in keys])

    print("\n  far 로 갔을 때 통계가 '사과 studio' 쪽으로 이동하는가(유클리드 거리):")
    for item in ("감자", "사과", "배", "감귤", "양파"):
        d_own = float(np.linalg.norm(vec(f"{item} far") - vec(f"{item} studio")))
        d_app = float(np.linalg.norm(vec(f"{item} far") - vec("사과 studio")))
        print(f"    {item} far → 자기 studio {d_own:6.1f} / 사과 studio {d_app:6.1f}")


# ── 3. 배경색 × 차지비율 교차 ─────────────────────────────────────────
def test_grid(picks: dict[str, list[Path]], items: list[str]) -> dict:
    print("\n" + "=" * 78)
    print("3. 배경색 × 차지비율 교차표 — '배경 탓'과 '크기 탓'을 가른다")
    print("=" * 78)

    out: dict = {}
    for item in items:
        imgs = load(picks[item])
        print(f"\n■ {item} (n={len(imgs)})   각 칸 = top-1 / top-3")
        head = f"{'배경':<20}" + "".join(f"{int(o * 100):>5}%   " for o in OCCUPANCIES) + "  배경만"
        print(head)
        print("-" * len(head))
        out[item] = {}
        for label, rgb in BACKGROUNDS.items():
            cells, rec = [], {}
            for occ in OCCUPANCIES:
                lg = logits([compose(im, rgb, occ) for im in imgs])
                t1, t3 = topk_rate(lg, item, 1), topk_rate(lg, item, 3)
                rec[occ] = (t1, t3)
                cells.append(f"{t1:>3.0%}/{t3:<3.0%} ")
            bl = logits([blank(rgb)])
            bpred = ITEMS[int(bl.argmax(1)[0])]
            bconf = float(softmax(bl)[0].max())
            out[item][label] = {"cells": rec, "blank": (bpred, bconf)}
            print(f"{label:<20}" + "".join(cells) + f"  {bpred}{bconf:.0%}")
    return out


def test_grid_summary(grid: dict) -> None:
    print("\n" + "-" * 78)
    print("배경/크기 분해 (감자 기준)")
    print("-" * 78)
    g = grid["감자"]
    for occ in OCCUPANCIES:
        vals = [g[b]["cells"][occ][0] for b in BACKGROUNDS]
        v3 = [g[b]["cells"][occ][1] for b in BACKGROUNDS]
        print(f"  차지 {int(occ*100):>3}% : top-1 배경별 {min(vals):.0%}~{max(vals):.0%} "
              f"(폭 {max(vals)-min(vals):.0%}p, 평균 {np.mean(vals):.0%}) | "
              f"top-3 {min(v3):.0%}~{max(v3):.0%} (평균 {np.mean(v3):.0%})")
    print("\n  같은 배경 안에서 크기만 바꿨을 때의 낙폭:")
    for b in BACKGROUNDS:
        c = g[b]["cells"]
        print(f"    {b:<20} 100%→15% : {c[1.0][0]:.0%} → {c[0.15][0]:.0%}  ({c[0.15][0]-c[1.0][0]:+.0%}p)")


# ── 4. 크기 말고 '디테일'만 죽이면 어떻게 되나 ────────────────────────
def test_detail(picks: dict[str, list[Path]], items: list[str]) -> None:
    print("\n" + "=" * 78)
    print("4. 프레임은 꽉 채우고 해상도만 낮추면? — '픽셀 부족' 가설 검증")
    print("   (far 에서 감자는 프레임의 25%, 즉 실효 56x56 픽셀만 차지한다)")
    print("=" * 78)
    ks = [224, 112, 56, 34, 22]
    head = f"{'품목':<8}" + "".join(f"{f'{k}px':>12}" for k in ks)
    print("\n" + head)
    print("-" * len(head))
    for item in items:
        imgs = load(picks[item])
        cells = []
        for k in ks:
            lg = logits([detail_only(im, k) for im in imgs])
            cells.append(f"{topk_rate(lg, item, 1):>4.0%}/{topk_rate(lg, item, 3):<4.0%}  ")
        print(f"{item:<8}" + "".join(cells))
    print("\n  ※ 56px 에서도 top-1 이 높으면 '픽셀이 없어서'가 아니다 → 배경이 섞여서다")


# ── 5. 배경만/피사체만 — 무엇이 사과 logit 을 밀어올리나 ──────────────
def test_ablation(picks: dict[str, list[Path]]) -> None:
    print("\n" + "=" * 78)
    print("5. 배경 단독 예측 — 빈 화면을 모델은 무엇이라 부르는가")
    print("=" * 78)
    print(f"\n{'배경':<20}" + "".join(f"{it:>9}" for it in ITEMS) + "   → 예측")
    for label, rgb in BACKGROUNDS.items():
        lg = logits([blank(rgb)])[0]
        pr = softmax(lg)
        print(f"{label:<20}" + "".join(f"{p:>9.1%}" for p in pr) + f"   → {ITEMS[int(lg.argmax())]}")

    print("\n  감자 far 에서 배경 비중을 서서히 올려보기(감자 원본 ↔ 회색 배경 알파 혼합):")
    imgs = load(picks["감자"])
    gray = BACKGROUNDS["회색(150,148,143)"]
    for a in (0.0, 0.25, 0.5, 0.75, 0.9):
        mixed = [Image.blend(im, blank(gray), a) for im in imgs]
        lg = logits(mixed)
        pred = [ITEMS[i] for i in lg.argmax(1)]
        print(f"    배경 섞음 {a:>4.0%} : 감자 top-1 {topk_rate(lg,'감자',1):>4.0%}  "
              f"top-3 {topk_rate(lg,'감자',3):>4.0%}  최빈예측 {max(set(pred), key=pred.count)}")


# ── 6. 작아졌을 때 '무엇으로' 틀리는가 ────────────────────────────────
def test_confusion(picks: dict[str, list[Path]]) -> None:
    print("\n" + "=" * 78)
    print("6. 회색 배경에서 작아질 때 예측이 어디로 몰리는가")
    print("=" * 78)
    gray = BACKGROUNDS["회색(150,148,143)"]
    for occ in (1.00, 0.50, 0.35, 0.25, 0.15):
        print(f"\n  [차지 {int(occ*100)}%]")
        for item in ITEMS:
            imgs = load(picks[item])
            lg = logits([compose(im, gray, occ) for im in imgs])
            pred = [ITEMS[i] for i in lg.argmax(1)]
            conf = softmax(lg).max(axis=1).mean()
            dist = "  ".join(f"{it} {pred.count(it)/len(pred):>4.0%}" for it in ITEMS if pred.count(it))
            print(f"    정답 {item:<4} → {dist:<52} 평균신뢰 {conf:.0%}")


# ── 7. 작은 '무엇'이든 사과가 되는가 (합성 도형) ──────────────────────
def test_blob(picks: dict[str, list[Path]]) -> None:
    print("\n" + "=" * 78)
    print("7. 회색 배경 위 단색 원반 — 내용 없는 작은 물체는 무엇으로 불리는가")
    print("=" * 78)
    from PIL import ImageDraw
    gray = BACKGROUNDS["회색(150,148,143)"]
    colors = {
        "감자색(196,180,136)": (196, 180, 136),
        "사과색(176,124,113)": (176, 124, 113),
        "빨강(200,40,40)": (200, 40, 40),
        "흰색(240,240,240)": (240, 240, 240),
        "회색(150,148,143)": (150, 148, 143),
    }
    print(f"\n{'원반 색':<22}" + "".join(f"{f'{int(o*100)}%':>16}" for o in (1.0, 0.5, 0.25, 0.15)))
    for label, c in colors.items():
        cells = []
        for occ in (1.0, 0.5, 0.25, 0.15):
            im = blank(gray)
            d = ImageDraw.Draw(im)
            r = 112 * occ
            d.ellipse([112 - r, 112 - r, 112 + r, 112 + r], fill=c)
            lg = logits([im])[0]
            cells.append(f"{ITEMS[int(lg.argmax())]} {softmax(lg).max():>4.0%}   ")
        print(f"{label:<22}" + "".join(f"{c:>16}" for c in cells))


# ── 8. 같은 픽셀 크기의 감자로 화면을 도배하면? ───────────────────────
def test_tile(picks: dict[str, list[Path]]) -> None:
    print("\n" + "=" * 78)
    print("8. 감자 조각의 '픽셀 크기'는 far 와 똑같이 두고 화면만 꽉 채우면?")
    print("   far(25%)=감자가 56px 이면서 화면의 6%만 덮음 → 타일은 56px 이면서 화면 100% 덮음")
    print("=" * 78)
    gray = BACKGROUNDS["회색(150,148,143)"]
    print(f"\n{'조건':<34}{'top-1':>8}{'top-3':>8}   최빈예측")
    for item in ("감자", "사과"):
        imgs = load(picks[item])
        for n in (1, 2, 4):  # n×n 타일 → 조각 크기 224/n
            tiled = []
            for im in imgs:
                cell = im.resize((224 // n, 224 // n), Image.LANCZOS)
                cv = Image.new("RGB", (224, 224), gray)
                for r in range(n):
                    for c in range(n):
                        cv.paste(cell, (c * (224 // n), r * (224 // n)))
                tiled.append(cv)
            lg = logits(tiled)
            pred = [ITEMS[i] for i in lg.argmax(1)]
            print(f"{f'{item} {n}x{n} 타일(조각 {224//n}px, 화면 100%)':<34}"
                  f"{topk_rate(lg, item, 1):>8.0%}{topk_rate(lg, item, 3):>8.0%}   "
                  f"{max(set(pred), key=pred.count)}")
        # 대조군: 같은 조각 크기지만 한 개만 (=far)
        one = [compose(im, gray, 0.25) for im in imgs]
        lg = logits(one)
        pred = [ITEMS[i] for i in lg.argmax(1)]
        print(f"{f'{item} 1개만 25%(조각 56px, 화면 6%)':<34}"
              f"{topk_rate(lg, item, 1):>8.0%}{topk_rate(lg, item, 3):>8.0%}   "
              f"{max(set(pred), key=pred.count)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    picks = E.sample_images(args.n)
    want = [s.strip() for s in args.only.split(",") if s.strip()]
    run = lambda k: (not want) or k in want  # noqa: E731

    if run("logits"):
        test_logits(picks)
    if run("stats"):
        test_stats(picks)
    grid = None
    if run("grid"):
        grid = test_grid(picks, ["감자", "사과", "배"])
        test_grid_summary(grid)
    if run("detail"):
        test_detail(picks, ITEMS)
    if run("ablation"):
        test_ablation(picks)
    if run("confusion"):
        test_confusion(picks)
    if run("blob"):
        test_blob(picks)
    if run("tile"):
        test_tile(picks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
