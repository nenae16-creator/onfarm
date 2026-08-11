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

LOCALFOOD_FARMS = 2256          # 한국경제 2026-05-27
LOCALFOOD_STORES = 12
LOCALFOOD_SALES_EOK = 271
NATIONAL_ELDERLY_RATE = 0.788   # 2025 농림어업총조사, 경영주 60대 이상


def read_kosis(path: Path) -> tuple[int, str]:
    """KOSIS 다운로드 CSV 에서 천안시 농가 수를 찾는다(인코딩·서식이 제각각이라 관대하게)."""
    raw = path.read_bytes()
    text = None
    for enc in ("cp949", "euc-kr", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise SystemExit("CSV 인코딩을 알 수 없습니다.")

    rows = list(csv.reader(io.StringIO(text)))
    for row in rows:
        joined = " ".join(row)
        if "천안" not in joined:
            continue
        # 같은 행의 숫자 중 가장 큰 값을 농가 수로 본다(연도 열이 섞여 있을 수 있다)
        nums = []
        for cell in row:
            c = cell.replace(",", "").strip()
            if re.fullmatch(r"\d+(\.\d+)?", c):
                v = float(c)
                if v > 100:            # 연도(2024 등)와 구분하기 위한 하한
                    nums.append(int(v))
        if nums:
            return max(nums), joined[:80]
    raise SystemExit("CSV 에서 '천안' 행을 찾지 못했습니다. 지역 필터를 확인하세요.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosis", type=Path, help="KOSIS DT_1EA1011 다운로드 CSV")
    ap.add_argument("--total", type=int, help="천안시 전체 농가 수를 직접 입력")
    args = ap.parse_args()

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
