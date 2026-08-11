"""
만든 발표 자료를 이미지로 펼쳐 눈으로 확인한다.

보지 않고 만든 슬라이드는 글자가 겹치거나 상자 밖으로 흘러나가도 알 수 없다.
PDF 를 페이지별 PNG 로 뽑아 실제 배치를 확인한다.

    python tools/preview_deck.py
    python tools/preview_deck.py --pdf outputs/ON-FARM_천안_기획서.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
args = sys.argv[1:]
pdf_path = Path(args[args.index("--pdf") + 1]) if "--pdf" in args else ROOT / "outputs" / "ON-FARM_천안_기획서.pdf"
out = ROOT / "outputs" / "preview"
out.mkdir(parents=True, exist_ok=True)

if not pdf_path.exists():
    raise SystemExit(f"PDF 가 없습니다: {pdf_path}\n먼저 python tools/build_deck.py 를 실행하세요.")

doc = fitz.open(pdf_path)
for i, page in enumerate(doc, start=1):
    page.get_pixmap(dpi=110).save(out / f"slide-{i:02d}.png")
print(f"{doc.page_count}장 → {out}")
