"""
천안 로컬푸드 미참여 농가 비율을 계산한다 — ON-FARM 제안의 핵심 근거.

    미참여율 = (천안 전체 농가 − 로컬푸드 참여 2,256) / 천안 전체 농가

주의(제안서에 그대로 옮길 것):
  '전체 농가'(농림어업조사, 경지 10a 이상 기준)와 '로컬푸드 참여 농가'(시 사업 집계)는
  조사 주체·기준·시점이 다르다. 정밀한 차집합이 아니라 **규모 감각**이며,
  그렇게 밝히지 않으면 데이터 이해 항목에서 감점된다.

사용:
    python tools/cheonan_gap.py --kosis data/cheonan/DT_1EA1011.csv
    python tools/cheonan_gap.py --total 8000        # 수치를 직접 아는 경우
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

LOCALFOOD_FARMS = 2256          # 한국경제 2026-05-27
LOCALFOOD_STORES = 12
LOCALFOOD_SALES_EOK = 271
NATIONAL_ELDERLY_RATE = 0.788   # 2025 농림어업총조사, 경영주 60대 이상


def read_kosis(path: Path) -> tuple[int, str]:
    """
    KOSIS 다운로드 CSV 에서 천안시 **농가 수(가구)** 를 찾는다.

    ★열 이름을 반드시 보고 고른다.
      처음에는 '행에서 가장 큰 숫자'를 농가 수로 삼았는데, 같은 행에 농가인구(21,433명)가
      함께 있어 그 값을 집어 미참여율이 89.5% 로 부풀려졌다. 헤더를 읽어 '농가 (가구)'
      열만 쓴다.
    """
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise SystemExit("CSV 인코딩을 알 수 없습니다.")

    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]

    # '농가'가 들어가되 '인구'·'오차'는 빠진 열을 찾는다
    col = None
    header = None
    for row in rows:
        for i, cell in enumerate(row):
            name = cell.strip()
            if "농가" in name and "인구" not in name and "오차" not in name:
                col, header = i, name
                break
        if col is not None:
            break
    if col is None:
        raise SystemExit(
            "'농가 (가구)' 열을 찾지 못했습니다. KOSIS 다운로드 시 항목에 '농가'가 포함됐는지 확인하세요."
        )

    for row in rows:
        if "천안" not in " ".join(row) or col >= len(row):
            continue
        cell = row[col].replace(",", "").strip()
        if not re.fullmatch(r"\d+(\.\d+)?", cell):
            continue
        value = int(float(cell))
        # 농가인구를 잘못 집었는지 자기점검: 천안 농가는 만 단위를 넘지 않는다
        if value > 100_000:
            raise SystemExit(f"'{header}' 열 값 {value:,} 이 비정상입니다. 열 선택을 확인하세요.")
        return value, f"{' '.join(c for c in row if c.strip())[:70]}  [열: {header}]"

    raise SystemExit("CSV 에서 '천안' 행을 찾지 못했습니다. 지역 필터를 확인하세요.")


def _self_test() -> None:
    """
    `python tools/cheonan_gap.py --self-test` — 열 선택이 옳은지 확인한다.
    농가인구(21,433)를 농가 수로 잘못 집었던 실수가 재발하지 않게 고정한다.
    """
    import tempfile

    sample = (
        '"행정구역별(1)",행정구역별(2),2024,2024,2024,2024,2024,2024\n'
        '"행정구역별(1)",행정구역별(2),농가 (가구),상대표준오차,농가인구 (명),'
        '상대표준오차(농가인구),농가인구(남) (명),농가인구(여) (명)\n'
        '"충청남도",천안시,9488,4.5,21433,5.0,10231,11202\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig") as fh:
        fh.write(sample)
        tmp = Path(fh.name)
    try:
        value, src = read_kosis(tmp)
        assert value == 9488, f"농가 수를 {value} 로 읽었다 — 21433(농가인구)를 집으면 안 된다"
        assert "농가 (가구)" in src, f"열 표기가 없다: {src}"
        print(f"✅ 열 선택 정상: {value:,} 가구 (농가인구 21,433 과 구분됨)")
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosis", type=Path, help="KOSIS DT_1EA1011 다운로드 CSV")
    ap.add_argument("--total", type=int, help="천안시 전체 농가 수를 직접 입력")
    ap.add_argument("--self-test", action="store_true", help="열 선택 로직 자체 검사")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return 0

    if args.total:
        total, src = args.total, "직접 입력"
    elif args.kosis and args.kosis.exists():
        total, src = read_kosis(args.kosis)
    else:
        print("천안시 전체 농가 수가 필요합니다.\n", file=sys.stderr)
        print("KOSIS → 국내통계 → 농림어업 → 농림어업조사 → 농업", file=sys.stderr)
        print("  → 「행정구역(시군구)별 농가, 농가인구」(DT_1EA1011)", file=sys.stderr)
        print("  → 지역: 충청남도 천안시 → CSV 다운로드\n", file=sys.stderr)
        print("받은 뒤:  python tools/cheonan_gap.py --kosis <파일>", file=sys.stderr)
        return 2

    outside = total - LOCALFOOD_FARMS
    rate = outside / total if total else 0
    elderly_outside = outside * NATIONAL_ELDERLY_RATE

    print(f"출처: {src}\n")
    print("=" * 54)
    print(f"천안시 전체 농가            {total:>8,} 가구")
    print(f"로컬푸드 참여 농가          {LOCALFOOD_FARMS:>8,} 가구  (직매장 {LOCALFOOD_STORES}곳, 매출 {LOCALFOOD_SALES_EOK}억)")
    print(f"로컬푸드 체계 밖 농가       {outside:>8,} 가구")
    print(f"→ 미참여율                {rate:>8.1%}")
    print("=" * 54)
    print(f"\n그중 경영주 60대 이상 추정   약 {elderly_outside:>7,.0f} 가구")
    print(f"  (전국 비율 {NATIONAL_ELDERLY_RATE:.1%} 적용 — 천안 실제 분포 확보 시 교체할 것)")

    print("\n제안서 문장 초안")
    print("-" * 54)
    print(
        f"천안시는 로컬푸드 직매장 {LOCALFOOD_STORES}곳에 농가 {LOCALFOOD_FARMS:,}곳이 참여해\n"
        f"연 {LOCALFOOD_SALES_EOK}억 원의 성과를 내고 있습니다. 다만 이는 전체 농가 {total:,}가구의\n"
        f"{1-rate:.1%}이며, 나머지 {rate:.1%}({outside:,}가구)는 이 체계 밖에 있습니다.\n"
        f"직매장은 농가가 직접 상품을 갖다 놓고 가격표를 붙이고 재고를 관리하는 구조라,\n"
        f"고령·소규모 농가일수록 참여가 어렵습니다. ON-FARM 은 이 공백을 겨냥합니다."
    )
    print("-" * 54)
    print("\n※ 두 수치는 조사 주체와 기준이 다릅니다. 정밀 차집합이 아니라 규모 감각으로 제시하고,")
    print("   발표에서도 그렇게 밝혀야 데이터 이해 항목에서 감점되지 않습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
