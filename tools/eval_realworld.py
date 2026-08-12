"""
"사과를 찍으면 사과라고 하는가" 를 실제로 잰다.

발표장에서 심사위원 앞에 놓고 찍는 사진은 학습 데이터와 다르다.
학습은 흰 배경 스튜디오 정사각 사진, 시연은 폰으로 찍은 4:3 사진에 배경이 있다.
그 차이가 정확도를 얼마나 떨어뜨리는지 숫자로 만든다.

    python tools/eval_realworld.py                      # 전체 조건 비교
    python tools/eval_realworld.py --n 40 --conditions studio,stretch43
    python tools/eval_realworld.py --self-test

브라우저가 하는 전처리를 그대로 흉내 낸다(public/js/features.js):
    canvas 224x224 에 drawImage(source, 0, 0, 224, 224)  ← 비율 무시하고 늘림
    RGB 를 mean/std 로 정규화 → NCHW
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "onfarm_qc.onnx"
META = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
VALID = ROOT / "data" / "onfarm_cv" / "valid"
SIZE = 224
ITEMS: list[str] = META["items"]
MEAN = np.array(META["normalize"]["mean"], dtype=np.float32)
STD = np.array(META["normalize"]["std"], dtype=np.float32)


# ── 브라우저 전처리 재현 ───────────────────────────────────────────────
def to_tensor(img: Image.Image) -> np.ndarray:
    """features.js 의 extractModelPixels 와 같은 일 — 비율을 지키지 않고 224x224 로 늘린다."""
    rgb = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return rgb.transpose(2, 0, 1)[None].astype(np.float32)


# ── 시연 조건 흉내 ─────────────────────────────────────────────────────
def cond_studio(img: Image.Image) -> Image.Image:
    """학습과 같은 조건 — 이 값이 상한이다."""
    return img


def cond_stretch43(img: Image.Image) -> Image.Image:
    """폰 가로 사진(4:3). 정사각 피사체가 가로로 늘어난 채 들어온다."""
    w, h = img.size
    canvas = Image.new("RGB", (int(h * 4 / 3), h), (245, 245, 245))
    canvas.paste(img, ((canvas.width - w) // 2, 0))
    return canvas


def cond_portrait34(img: Image.Image) -> Image.Image:
    """폰 세로 사진(3:4). 실제 농가가 가장 많이 찍는 모양."""
    w, h = img.size
    canvas = Image.new("RGB", (w, int(w * 4 / 3)), (245, 245, 245))
    canvas.paste(img, (0, (canvas.height - h) // 2))
    return canvas


def cond_background(img: Image.Image) -> Image.Image:
    """흰 배경이 아닌 곳(마당·상자·흙) 위에 놓고 찍은 상황."""
    w, h = img.size
    bg = Image.new("RGB", (int(w * 1.35), int(h * 1.35)), (122, 104, 78))
    noise = np.random.default_rng(0).integers(-18, 18, (bg.height, bg.width, 3), dtype=np.int16)
    bg = Image.fromarray(np.clip(np.asarray(bg, np.int16) + noise, 0, 255).astype(np.uint8))
    bg.paste(img, ((bg.width - w) // 2, (bg.height - h) // 2))
    return bg


def cond_far(img: Image.Image) -> Image.Image:
    """멀리서 찍어 피사체가 화면의 일부만 차지하는 경우."""
    w, h = img.size
    bg = Image.new("RGB", (w * 2, h * 2), (150, 148, 143))
    small = img.resize((w // 2, h // 2), Image.LANCZOS)
    bg.paste(small, ((bg.width - small.width) // 2, (bg.height - small.height) // 2))
    return bg


def cond_dim(img: Image.Image) -> Image.Image:
    """실내·그늘. 밝기가 떨어진다."""
    return ImageEnhance.Brightness(img).enhance(0.55)


def cond_tilt(img: Image.Image) -> Image.Image:
    """손으로 들고 찍어 기울어진 경우."""
    return img.rotate(12, resample=Image.BILINEAR, expand=True, fillcolor=(245, 245, 245))


CONDITIONS = {
    "studio": ("학습과 같은 조건", cond_studio),
    "stretch43": ("폰 가로 4:3", cond_stretch43),
    "portrait34": ("폰 세로 3:4", cond_portrait34),
    "background": ("흙·상자 배경", cond_background),
    "far": ("멀리서 촬영", cond_far),
    "dim": ("어두운 곳", cond_dim),
    "tilt": ("기울어짐", cond_tilt),
}


def sample_images(n: int, seed: int = 7) -> dict[str, list[Path]]:
    """품목별로 검증 분할에서 고르게 뽑는다. 같은 개체가 몰리지 않게 파일명을 섞는다."""
    rng = random.Random(seed)
    out: dict[str, list[Path]] = {}
    for item in ITEMS:
        base = VALID / item
        if not base.exists():
            continue
        files = [p for g in base.iterdir() if g.is_dir() for p in g.glob("*.jpg")]
        rng.shuffle(files)
        out[item] = files[:n]
    return out


def run(session: ort.InferenceSession, batch: list[np.ndarray]) -> np.ndarray:
    x = np.concatenate(batch, axis=0)
    outputs = session.run(None, {session.get_inputs()[0].name: x})
    names = [o.name for o in session.get_outputs()]
    logits = outputs[names.index("item_logits")] if "item_logits" in names else outputs[0]
    return np.asarray(logits)


def _self_test() -> None:
    """조건 함수들이 실제로 이미지를 바꾸는지 — 아무것도 안 하면 측정이 헛돈다."""
    img = Image.new("RGB", (200, 200), (200, 100, 50))
    for key, (label, fn) in CONDITIONS.items():
        got = fn(img)
        if key == "studio":
            assert got.size == img.size, "studio 는 원본 그대로여야 한다"
            continue
        changed = got.size != img.size or np.asarray(got.resize((32, 32))).astype(int).sum() != \
            np.asarray(img.resize((32, 32))).astype(int).sum()
        assert changed, f"'{label}' 조건이 이미지를 전혀 바꾸지 않는다"
    t = to_tensor(img)
    assert t.shape == (1, 3, SIZE, SIZE), t.shape
    print(f"✅ 조건 {len(CONDITIONS)}종이 모두 이미지를 바꾸고, 텐서는 {t.shape}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="품목당 표본 수")
    ap.add_argument("--conditions", default="", help="쉼표로 구분(기본: 전체)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", type=Path, help="결과를 JSON 으로 저장")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return 0

    if not MODEL.exists():
        raise SystemExit(f"모델이 없습니다: {MODEL}")
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    picks = sample_images(args.n)
    if not picks:
        raise SystemExit(f"검증 이미지가 없습니다: {VALID}")

    keys = [k.strip() for k in args.conditions.split(",") if k.strip()] or list(CONDITIONS)
    results: dict[str, dict[str, float]] = {}

    for key in keys:
        label, fn = CONDITIONS[key]
        per_item: dict[str, float] = {}
        for item, files in picks.items():
            if not files:
                continue
            tensors = [to_tensor(fn(Image.open(f))) for f in files]
            logits = run(session, tensors)
            pred = [ITEMS[i] for i in logits.argmax(axis=1)]
            per_item[item] = sum(p == item for p in pred) / len(pred)
        overall = sum(per_item.values()) / len(per_item)
        results[key] = {"label": label, "overall": overall, **per_item}

    width = max(len(v["label"]) for v in results.values()) + 2
    print(f"\n검증 분할 이미지 · 품목당 {args.n}장 · 브라우저 전처리(224x224 늘리기) 재현\n")
    header = f"{'조건':<{width}}{'전체':>7}  " + "  ".join(f"{i:>5}" for i in ITEMS)
    print(header)
    print("-" * len(header))
    for key in keys:
        r = results[key]
        cells = "  ".join(f"{r.get(i, 0):>5.0%}" for i in ITEMS)
        mark = "  ←기준" if key == "studio" else ""
        print(f"{r['label']:<{width}}{r['overall']:>7.1%}  {cells}{mark}")

    base = results.get("studio", {}).get("overall")
    if base:
        worst = min((k for k in results if k != "studio"),
                    key=lambda k: results[k]["overall"], default=None)
        if worst:
            w = results[worst]
            print(f"\n가장 크게 떨어지는 조건: {w['label']}  "
                  f"{base:.1%} → {w['overall']:.1%}  ({w['overall'] - base:+.1%}p)")

    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
