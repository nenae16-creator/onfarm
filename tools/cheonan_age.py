"""
천안 농가 경영주의 연령과 경지규모를 **천안 자체 원자료에서** 계산한다.

지금까지는 전국 비율(78.8%)을 천안에 그대로 적용한 추정치를 썼다.
그 값은 출처를 확인하지 못했고, 같은 표에서 계산한 전국 값과도 맞지 않았다.
KOSIS OpenAPI 로 천안시(코드 34010) 원자료를 받았으므로 추정을 걷어낸다.

    python tools/cheonan_age.py              # data/cheonan/ 의 저장본으로 계산
    python tools/cheonan_age.py --self-test

원자료 재수집:
    python tools/kosis_api.py fetch DT_1AG20107 --prd F --obj1 34010 --obj2 ALL
    python tools/kosis_api.py fetch DT_1AG20107 --prd F --obj1 00    --obj2 ALL
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cheonan"
CHEONAN_CSV = DATA / "KOSIS_DT_1AG20107_천안.csv"
NATION_CSV = DATA / "KOSIS_DT_1AG20107_00.csv"

OLD_BANDS = ("60~64", "65~69", "70~74", "75~79", "80세이상")
SMALL_BANDS = ("0.1ha 미만", "0.1~0.2", "0.2~0.3", "0.3~0.5", "0.5~0.7", "0.7~1.0")

# 로컬푸드 미참여 농가(2024 기준). tools/cheonan_gap.py 와 같은 값이어야 한다.
OUTSIDE_2024 = 7_232


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"원자료가 없습니다: {path.name}\n"
            "python tools/kosis_api.py fetch DT_1AG20107 --prd F --obj1 34010 --obj2 ALL"
        )
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def value(rows: list[dict], item: str, group: str = "계") -> float:
    """항목 이름으로 찾는다 — 순서나 위치로 집지 않는다."""
    for r in rows:
        if r.get("분류") == group and r.get("항목", "").strip() == item:
            raw = (r.get("값") or "").replace(",", "").strip()
            if raw in ("", "-", "X"):
                return 0.0
            return float(raw)
    return 0.0


def age_profile(rows: list[dict], label: str) -> dict:
    total = value(rows, "농가")
    if total <= 0:
        raise SystemExit(f"{label}: 전체 농가 수를 읽지 못했습니다.")
    old = sum(value(rows, b) for b in OLD_BANDS)
    mean = value(rows, "경영주평균연령")
    return {"label": label, "total": total, "old": old, "rate": old / total, "mean": mean}


def farmland(rows: list[dict]) -> dict:
    total = value(rows, "농가")
    small = sum(value(rows, "농가", group=b) for b in SMALL_BANDS)
    return {"total": total, "small": small, "rate": small / total if total else 0}


def _self_test() -> None:
    """항목 이름으로 고르는지, 다른 값에 딸려 들어가지 않는지 고정한다."""
    rows = [
        {"분류": "계", "항목": "농가", "값": "11111"},
        {"분류": "계", "항목": "60~64", "값": "2097"},
        {"분류": "계", "항목": "65~69", "값": "1826"},
        {"분류": "계", "항목": "70~74", "값": "1591"},
        {"분류": "계", "항목": "75~79", "값": "1181"},
        {"분류": "계", "항목": "80세이상", "값": "1078"},
        {"분류": "계", "항목": "경영주평균연령", "값": "65.1"},
        # 함정: 다른 분류에도 같은 이름이 있다. '계' 만 읽어야 한다.
        {"분류": "논벼", "항목": "농가", "값": "9999"},
        {"분류": "논벼", "항목": "60~64", "값": "9999"},
    ]
    p = age_profile(rows, "테스트")
    assert p["total"] == 11111, f"전체를 {p['total']} 로 읽었다 — 다른 분류가 섞였다"
    assert p["old"] == 7773, f"60대 이상을 {p['old']} 로 읽었다"
    assert abs(p["rate"] - 0.6996) < 0.001, f"비율 {p['rate']}"
    assert p["mean"] == 65.1
    print(f"✅ 항목 이름으로 정확히 집는다: 11,111 중 7,773 = {p['rate']:.1%}, 평균 {p['mean']}세")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return 0

    ch = load(CHEONAN_CSV)
    cheonan = age_profile(ch, "천안시")
    land = farmland(ch)
    year = ch[0].get("시점", "?")
    table = ch[0].get("표이름", "?")

    print(f"출처: KOSIS DT_1AG20107 「{table}」 {year}년 (농림어업총조사, 전수)\n")
    print("=" * 58)
    print(f"천안시 농가                 {cheonan['total']:>9,.0f} 가구")
    print(f"경영주 평균 연령            {cheonan['mean']:>9.1f} 세")
    print(f"경영주 60대 이상            {cheonan['old']:>9,.0f} 가구   {cheonan['rate']:>6.1%}")
    print(f"경지 1ha 미만               {land['small']:>9,.0f} 가구   {land['rate']:>6.1%}")
    print("=" * 58)

    if NATION_CSV.exists():
        nation = age_profile(load(NATION_CSV), "전국")
        print(f"\n같은 조사 전국   농가 {nation['total']:,.0f} · 평균 {nation['mean']:.1f}세 · "
              f"60대 이상 {nation['rate']:.1%}")
        print(f"→ 천안은 전국보다 {nation['rate'] - cheonan['rate']:+.1%}p "
              f"({'젊다' if cheonan['rate'] < nation['rate'] else '고령이다'})")

    est = OUTSIDE_2024 * cheonan["rate"]
    print(f"\n로컬푸드 체계 밖 {OUTSIDE_2024:,}가구에 천안 비율 {cheonan['rate']:.1%} 적용")
    print(f"→ 그중 경영주 60대 이상 약 {est:,.0f}가구")
    print("\n※ 전제: 로컬푸드 참여 여부와 경영주 연령이 무관하다고 본 것이다.")
    print("   실제로는 참여 농가가 더 젊을 가능성이 높아 이 값은 보수적인 하한이다.")
    print(f"※ 연도가 다르다. 농가 수는 2024년(9,488), 연령 비율은 {year}년 총조사다.")
    print("   발표에서 이 두 가지를 먼저 밝힌다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
