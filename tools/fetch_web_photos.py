"""
웹 공개 사진을 품목별로 받아 평가셋을 만든다 — 출처·라이선스를 함께 기록한다.

★이것은 '실환경 평가'가 아니다.
  웹 사진은 대부분 상품컷·전시컷이라 오히려 학습 데이터(스튜디오)에 가깝다.
  고령 농가가 밭에서 찍은 폰 사진과는 다르다. 그래서 이 결과로
  models/metadata.json 의 field_evaluated 를 true 로 바꾸지 않는다.
  '스튜디오와 실사진 사이의 제3의 조건' 으로만 쓴다.

★라이선스가 분명한 곳에서만 받는다.
  공고문이 "제3자의 라이선스를 침해하지 말 것"을 명시한다.
  위키미디어 커먼즈는 파일마다 라이선스가 API 로 확인되므로 그것만 쓴다.
  받은 파일마다 출처 URL·저작자·라이선스를 CSV 로 남긴다.

    python tools/fetch_web_photos.py                 # 품목당 6장
    python tools/fetch_web_photos.py --per-item 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "web_photos"
API = "https://commons.wikimedia.org/w/api.php"
UA = "ON-FARM-contest-research/1.0 (https://github.com/nenae16-creator/onfarm)"

# 품목별 커먼즈 분류.
# ★텍스트 검색은 못 쓴다. "apple" 로 찾으면 애플II 컴퓨터가, "pear" 로 찾으면 배꽃이 나온다.
#   실제로 처음 받은 30장 중 12장이 그런 식으로 오염됐다(나방·밭 풍경·도해 그림 포함).
#   분류는 사람이 분류해 둔 것이라 훨씬 정확하다. 그래도 마지막엔 눈으로 확인한다.
CATEGORIES: dict[str, list[str]] = {
    "사과": ["Category:Apples"],
    "배": ["Category:Pyrus pyrifolia", "Category:Nashi"],
    "감귤": ["Category:Citrus unshiu", "Category:Mandarin oranges",
            "Category:Citrus reticulata"],
    "감자": ["Category:Potatoes"],
    "양파": ["Category:Onions"],
}

# 분류 안에도 섞여 드는 것들 — 제목으로 걸러 낸다
TITLE_BLOCK = ("flower", "blossom", "tree", "leaf", "leaves", "seedling", "diagram",
               "moth", "larva", "disease", "field", "orchard", "plant", "bark",
               "macintosh", "iigs", "computer", "logo", "map", "chart", "slice",
               "cut ", "sliced", "cross section", "juice", "pie", "cake", "sauce")

# 재배포·재사용이 자유로운 것만 받는다
ALLOWED = ("cc0", "publicdomain", "cc-by", "cc by")


def call(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as res:
        body = json.loads(res.read().decode("utf-8"))
    time.sleep(1.2)   # 커먼즈 API 는 빠르게 부르면 429 를 준다
    return body


def category_files(category: str, limit: int = 60) -> list[str]:
    """분류에 실제로 들어 있는 파일만 가져온다."""
    data = call({
        "action": "query", "list": "categorymembers", "cmtitle": category,
        "cmtype": "file", "cmlimit": str(limit),
    })
    return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]


def title_ok(title: str) -> bool:
    low = title.lower()
    return not any(b in low for b in TITLE_BLOCK)


def image_info(titles: list[str]) -> dict[str, dict]:
    """한 번에 여러 파일의 URL·라이선스·저작자를 가져온다."""
    out: dict[str, dict] = {}
    for i in range(0, len(titles), 20):
        data = call({
            "action": "query", "titles": "|".join(titles[i:i + 20]),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": "1024",
        })
        for page in data.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            out[page.get("title", "")] = {
                "url": info.get("thumburl") or info.get("url"),
                "descriptionurl": info.get("descriptionurl", ""),
                "license": (meta.get("LicenseShortName", {}) or {}).get("value", ""),
                "artist": (meta.get("Artist", {}) or {}).get("value", ""),
                "width": info.get("width", 0),
                "height": info.get("height", 0),
            }
    return out


def strip_html(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s).replace("\n", " ").strip()[:80]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-item", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for item, cats in CATEGORIES.items():
        folder = OUT / item
        for old in folder.glob("*"):
            old.unlink()
        folder.mkdir(exist_ok=True)
        got = 0
        seen: set[str] = set()

        for cat in cats:
            if got >= args.per_item:
                break
            titles = [t for t in category_files(cat) if t not in seen and title_ok(t)]
            seen.update(titles)
            info = image_info(titles)

            for title, meta in info.items():
                if got >= args.per_item:
                    break
                lic = (meta.get("license") or "").lower()
                if not any(a in lic for a in ALLOWED):
                    continue                       # 라이선스가 불명확하면 받지 않는다
                if not meta.get("url"):
                    continue
                if min(meta.get("width", 0), meta.get("height", 0)) < 400:
                    continue                       # 너무 작으면 평가에 부적절

                ext = Path(urllib.parse.urlparse(meta["url"]).path).suffix or ".jpg"
                dest = folder / f"{item}_{got + 1:02d}{ext}"
                try:
                    req = urllib.request.Request(meta["url"], headers={"User-Agent": UA})
                    with urllib.request.urlopen(req, timeout=60) as res:
                        dest.write_bytes(res.read())
                except Exception as err:
                    print(f"  건너뜀 {title}: {err}")
                    continue

                rows.append({
                    "품목": item, "파일": str(dest.relative_to(ROOT)),
                    "원본제목": title, "출처": meta.get("descriptionurl", ""),
                    "라이선스": meta.get("license", ""),
                    "저작자": strip_html(meta.get("artist", "")),
                })
                got += 1
                time.sleep(0.6)

        print(f"  {item}: {got}장")

    manifest = OUT / "출처_라이선스.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["품목", "파일", "원본제목", "출처", "라이선스", "저작자"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n총 {len(rows)}장 · 출처 기록: {manifest.relative_to(ROOT)}")
    print("★이 사진들은 실환경(농가 폰) 사진이 아니다. field_evaluated 를 바꾸지 않는다.")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
