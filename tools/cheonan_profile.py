"""
천안 농가의 '누구인가'(경영주 연령)와 '얼마나 작은가'(경지규모)를 KOSIS CSV 에서 읽는다.

지금 제안서는 경영주 60대 이상 비율에 **전국 값 78.8%** 를 그대로 적용한 추정치를 쓴다.
천안 실제 분포를 넣으면 그 추정 표기를 지울 수 있다.

    python tools/cheonan_profile.py                       # data/cheonan/ 의 CSV 전부 읽는다
    python tools/cheonan_profile.py --csv <파일>
    python tools/cheonan_profile.py --self-test           # 열 선택 로직 검사

★열은 반드시 헤더 이름으로 고른다.
  전에 '행에서 가장 큰 숫자' 를 농가 수로 집었다가 농가인구(21,433)를 읽어
  미참여율이 76.2% 대신 89.5% 로 부풀려진 적이 있다. 같은 실수를 막는 게 이 파일의 절반이다.
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cheonan"

CHEONAN_FARMS = 9_488       # DT_1EA1011(2024) — 대조용 기준값
NATIONAL_ELDERLY = 0.788    # 2025 농림어업총조사 전국 경영주 60대 이상
REGION = "천안"

# 값이 아니라 설명인 열 — 합계 계산에서 뺀다
NOISE = re.compile(r"오차|비율|구성비|증감|전년|지수")


def decode(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"인코딩을 알 수 없습니다: {path.name}")


def read_region_row(text: str) -> tuple[list[str], list[str], str]:
    """
    KOSIS 다운로드 CSV 는 머리글이 여러 줄이다(시점 줄 + 항목 줄).
    항목 줄과 천안 행을 찾아 (항목명, 값) 을 짝지어 돌려준다.
    """
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        raise SystemExit("빈 CSV 입니다.")

    target = next((r for r in rows if any(REGION in c for c in r)), None)
    if target is None:
        raise SystemExit(
            f"'{REGION}' 행이 없습니다. KOSIS 다운로드 시 지역을 충청남도 천안시로 골랐는지 확인하세요."
        )

    # 항목 줄 = 천안 행 바로 위쪽 줄 중, 숫자가 아닌 값이 가장 많은 줄
    above = rows[: rows.index(target)]
    if not above:
        raise SystemExit("머리글 줄을 찾지 못했습니다.")

    def wordiness(row: list[str]) -> int:
        return sum(1 for c in row if c.strip() and not re.fullmatch(r"[\d,.\s]+", c))

    header = max(above, key=wordiness)
    width = min(len(header), len(target))
    return header[:width], target[:width], " ".join(c for c in target if c.strip())[:80]


def numeric(pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
    out = []
    for name, raw in pairs:
        cell = raw.replace(",", "").strip()
        if not re.fullmatch(r"-?\d+(\.\d+)?", cell):
            continue
        if NOISE.search(name):
            continue
        out.append((name.strip(), float(cell)))
    return out


def classify(names: list[str]) -> str:
    joined = " ".join(names)
    if re.search(r"\d+\s*세|세\s*미만|세\s*이상|\d+\s*[~～-]\s*\d+", joined):
        return "age"
    if re.search(r"ha|헥타|㎡|평|규모", joined):
        return "farmland"
    return "unknown"


def report(path: Path) -> None:
    header, row, src = read_region_row(decode(path))
    cols = numeric(list(zip(header, row)))
    if not cols:
        raise SystemExit(f"{path.name}: 숫자 열을 찾지 못했습니다.")

    kind = classify([n for n, _ in cols])
    print(f"\n■ {path.name}")
    print(f"  원본 행: {src}")
    print(f"  형태: {'경영주 연령별' if kind == 'age' else '경지규모별' if kind == 'farmland' else '판별 실패'}")

    total = next((v for n, v in cols if n in ("계", "합계", "소계", "전체")), None)
    parts = [(n, v) for n, v in cols if n not in ("계", "합계", "소계", "전체")]
    base = total if total is not None else sum(v for _, v in parts)

    # ★농가인구를 농가 수로 잘못 읽었는지 확인한다 — 이 검사가 89.5% 사고를 막는다
    if base > CHEONAN_FARMS * 1.5:
        print(f"  ⚠ 합계 {base:,.0f} 이 천안 농가 {CHEONAN_FARMS:,} 보다 크게 많습니다.")
        print("    농가인구(명) 열을 읽었을 수 있습니다. KOSIS 항목에서 '농가(가구)' 를 골랐는지 확인하세요.")
    elif abs(base - CHEONAN_FARMS) / CHEONAN_FARMS > 0.15:
        print(f"  ⚠ 합계 {base:,.0f} 이 기준값 {CHEONAN_FARMS:,} 과 15% 넘게 차이납니다(조사·연도 차이일 수 있음).")

    print()
    for name, value in cols:
        share = value / base if base else 0
        print(f"    {name:<22} {value:>10,.0f}   {share:>6.1%}")

    if kind == "age":
        old = sum(v for n, v in parts if re.search(r"(?:^|\D)(6\d|7\d|8\d)(?:\s*세|\s*[~～-]|\D|$)", n))
        if old:
            rate = old / base
            print(f"\n  경영주 60대 이상  {old:,.0f}가구  ({rate:.1%})")
            print(f"  전국 {NATIONAL_ELDERLY:.1%} 대비 {rate - NATIONAL_ELDERLY:+.1%}p")
            print("\n  → 제안서에서 '전국 비율을 적용한 추정' 표기를 지우고 이 값을 쓸 수 있습니다.")
            print(f"     docs/CHEONAN_EVIDENCE.md 와 tools/cheonan_gap.py 의 "
                  f"NATIONAL_ELDERLY_RATE 를 함께 고치세요.")
    elif kind == "farmland":
        small = sum(v for n, v in parts if re.search(r"0\.5|5천|미만", n))
        if small:
            print(f"\n  소규모(하위 구간 합)  {small:,.0f}가구  ({small / base:.1%})")
            print("  → '소농' 을 수치로 말할 수 있게 됩니다. 지금은 근거가 없어 말하지 않고 있습니다.")


def _self_test() -> None:
    """농가인구를 농가 수로 착각하는 실수가 재발하지 않는지 고정한다."""
    age = (
        '"행정구역별(1)",행정구역별(2),2024,2024,2024,2024\n'
        '"행정구역별(1)",행정구역별(2),계,40세 미만,40~59세,60세 이상\n'
        '"충청남도",천안시,9488,104,2100,7284\n'
    )
    header, row, _ = read_region_row(age)
    cols = numeric(list(zip(header, row)))
    names = [n for n, _ in cols]
    assert classify(names) == "age", f"연령별로 판별하지 못했다: {names}"
    total = dict(cols)["계"]
    assert total == 9488, f"계를 {total} 로 읽었다"

    trap = (
        '"행정구역별(1)",행정구역별(2),2024,2024\n'
        '"행정구역별(1)",행정구역별(2),농가 (가구),농가인구 (명)\n'
        '"충청남도",천안시,9488,21433\n'
    )
    header, row, _ = read_region_row(trap)
    cols = numeric(list(zip(header, row)))
    assert dict(cols)["농가 (가구)"] == 9488
    assert dict(cols)["농가인구 (명)"] == 21433, "농가인구 열이 사라졌다 — 구분이 안 된다"
    print("✅ 열 선택 정상: 계 9,488 / 농가인구 21,433 을 서로 다른 값으로 읽는다")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, action="append", help="읽을 CSV(여러 번 지정 가능)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return 0

    files = args.csv or sorted(p for p in DATA.glob("*.csv") if "1EA1011" not in p.name)
    if not files:
        print("읽을 CSV 가 없습니다.\n", file=sys.stderr)
        print("KOSIS 농림어업총조사(전수)에서 아래 두 가지를 받아 data/cheonan/ 에 넣으세요.", file=sys.stderr)
        print("  · 경영주 연령 및 교육정도별 농가   DT_1AG20107 (2020)", file=sys.stderr)
        print("  · 경지규모별 농가수 및 경지면적   (농림어업총조사 2020 → 시군구 단위)", file=sys.stderr)
        print("※ DT_1EA1015/DT_1EA1019는 농림어업조사 표본표로 시도까지만 제공합니다.", file=sys.stderr)
        print("자세한 경로는 docs/CHEONAN_EVIDENCE.md 참고", file=sys.stderr)
        return 2

    for f in files:
        report(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
