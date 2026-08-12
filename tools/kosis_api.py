"""
KOSIS OpenAPI 로 통계표를 확인하고 천안 행만 뽑아 온다.

브라우저로 CSV 를 받는 것보다 이쪽이 낫다 — 표 안에 천안이 실제로 있는지,
항목 이름이 무엇인지를 **받기 전에** 확인할 수 있다.

    python tools/kosis_api.py probe DT_1AG20107     # 이 표에 뭐가 들어 있나
    python tools/kosis_api.py fetch DT_1EA1015      # 천안 행을 CSV 로 저장

인증키는 무료로 발급받는다(즉시 발급):
    https://kosis.kr/openapi/index/index.jsp  →  활용신청

    $env:KOSIS_API_KEY = "발급받은키"      (PowerShell)
    python tools/kosis_api.py probe DT_1EA1015

★키를 코드나 문서에 적어 두지 말 것. 환경변수로만 넘긴다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cheonan"
ENDPOINT = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
SIGNUP = "https://kosis.kr/openapi/index/index.jsp"
REGION = "천안"


def call(key: str, tbl: str, *, obj_l1="ALL", obj_l2="", itm="ALL",
         prd="Y", count=1, org="101") -> list[dict]:
    params = {
        "method": "getList", "apiKey": key, "orgId": org, "tblId": tbl,
        "itmId": itm, "objL1": obj_l1, "format": "json", "jsonVD": "Y",
        "prdSe": prd, "newEstPrdCnt": str(count),
    }
    if obj_l2:
        params["objL2"] = obj_l2
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=40) as res:
        body = json.loads(res.read().decode("utf-8"))

    if isinstance(body, dict):
        err, msg = body.get("err"), body.get("errMsg", "")
        if err == "11":
            raise SystemExit(
                f"인증키가 유효하지 않습니다 ({msg}).\n"
                f"무료 발급: {SIGNUP} → 활용신청\n"
                '발급 후:  $env:KOSIS_API_KEY = "받은키"'
            )
        raise SystemExit(f"KOSIS 응답 오류 {err}: {msg}")
    return body


def probe(key: str, tbl: str, prd: str) -> None:
    rows = call(key, tbl, prd=prd)
    if not rows:
        print("데이터가 비어 있습니다. 주기(--prd)를 바꿔 보세요(Y 연간 / F 부정기 / M 월).")
        return

    name = rows[0].get("TBL_NM", "(이름 없음)")
    period = sorted({r.get("PRD_DE", "") for r in rows})
    print(f"\n표 이름 : {name}")
    print(f"표 번호 : {tbl}")
    print(f"시점    : {', '.join(period)}")
    print(f"행 수   : {len(rows):,}")

    regions = sorted({r.get("C1_NM", "") for r in rows if r.get("C1_NM")})
    items = sorted({r.get("ITM_NM", "") for r in rows if r.get("ITM_NM")})
    cats = sorted({r.get("C2_NM", "") for r in rows if r.get("C2_NM")})

    hit = [r for r in regions if REGION in r]
    print(f"\n지역 {len(regions)}종 — {'· '.join(regions[:12])}{' …' if len(regions) > 12 else ''}")
    if hit:
        print(f"  ✅ '{REGION}' 있음 → {', '.join(hit)}")
    else:
        print(f"  ❌ '{REGION}' 없음 — 이 표는 시군구 단위가 아닙니다(시도까지만).")

    print(f"\n항목 {len(items)}종 — {'· '.join(items[:14])}{' …' if len(items) > 14 else ''}")
    if cats:
        print(f"\n분류2 {len(cats)}종 — {'· '.join(cats[:14])}{' …' if len(cats) > 14 else ''}")

    age = [x for x in items + cats if "세" in x]
    if age:
        print(f"\n  연령 구분으로 보이는 항목: {', '.join(age[:10])}")


def fetch(key: str, tbl: str, prd: str) -> None:
    rows = call(key, tbl, prd=prd)
    picked = [r for r in rows if REGION in (r.get("C1_NM") or "")]
    if not picked:
        raise SystemExit(
            f"'{REGION}' 행이 없습니다. 먼저 probe 로 이 표에 천안이 있는지 확인하세요."
        )

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"KOSIS_{tbl}_천안.csv"
    cols = ["TBL_NM", "PRD_DE", "C1_NM", "C2_NM", "ITM_NM", "DT", "UNIT_NM"]
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["표이름", "시점", "지역", "분류", "항목", "값", "단위"])
        for r in picked:
            w.writerow([r.get(c, "") for c in cols])

    print(f"✔ {out.relative_to(ROOT)}  ({len(picked)}행)")
    print(f"  표: {picked[0].get('TBL_NM', '')}")
    for r in picked[:12]:
        print(f"    {r.get('C2_NM') or r.get('ITM_NM'):<22} {r.get('DT'):>10} {r.get('UNIT_NM', '')}")
    if len(picked) > 12:
        print(f"    … 외 {len(picked) - 12}행")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["probe", "fetch"])
    ap.add_argument("tblId")
    ap.add_argument("--prd", default="Y", help="주기: Y 연간(기본) / F 부정기 / M 월")
    ap.add_argument("--key", default=os.environ.get("KOSIS_API_KEY", ""))
    args = ap.parse_args()

    if not args.key:
        print("인증키가 없습니다.\n", file=sys.stderr)
        print(f"무료 발급(즉시): {SIGNUP} → 활용신청", file=sys.stderr)
        print('발급 후:  $env:KOSIS_API_KEY = "받은키"', file=sys.stderr)
        return 2

    (probe if args.action == "probe" else fetch)(args.key, args.tblId, args.prd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
