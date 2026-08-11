"""
발표 대본의 낭독 시간을 **실제 음성 합성으로 잰다.**

글자 수로 추정하면 틀린다. 한국어는 숫자·영문·조사에 따라 발화 길이가 크게 달라서,
"대충 이 정도면 10분" 이라고 계산해 두고 현장에서 시간이 넘치는 일이 반복된다.
그래서 edge-tts 로 실제 음성을 만들고 ffprobe 로 길이를 잰다.

    python tools/measure_script.py
    python tools/measure_script.py --section 5      # 특정 절만
    python tools/measure_script.py --rate +10%      # 빠르게 읽을 때

`▸` 로 시작하는 줄(지시문)과 코드/표는 시간에서 뺀다.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "docs" / "PITCH_10MIN.md"
VOICE = "ko-KR-InJoonNeural"
TARGET_SEC = 600  # 10분


def find_ffprobe() -> str:
    for name in ("ffprobe", "ffprobe.exe"):
        if (p := shutil.which(name)):
            return p
    # winget 설치본은 PATH 에 없을 때가 있다. 디스크 전체를 훑지 않도록 경로를 좁힌다.
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    for p in winget.glob("Gyan.FFmpeg*/**/bin/ffprobe.exe"):
        return str(p)
    raise SystemExit("ffprobe 를 찾지 못했습니다. ffmpeg 를 설치하세요.")


def parse(md: str) -> list[tuple[str, str]]:
    """(절 제목, 낭독할 본문) 목록. 지시문·표·머리말은 뺀다."""
    sections: list[tuple[str, list[str]]] = []
    in_qa = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            title = line[3:].strip()
            in_qa = "Q&A" in title
            sections.append((title, []))
            continue
        if not sections or in_qa:
            continue
        if (
            not line.strip()
            or line.startswith("▸")          # 지시문
            or line.startswith("#")
            or line.startswith("|")
            or line.startswith("---")
            or line.startswith("```")
            or line.startswith("- ")
        ):
            continue
        # 굵게/기울임 표시는 발화에 영향이 없으니 지운다
        sections[-1][1].append(re.sub(r"[*`]", "", line).strip())
    return [(t, " ".join(body)) for t, body in sections if " ".join(body).strip()]


async def synth(text: str, out: Path, rate: str) -> None:
    import edge_tts

    await edge_tts.Communicate(text, VOICE, rate=rate).save(str(out))


def duration(path: Path, ffprobe: str) -> float:
    res = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(res.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="+0%", help="낭독 속도 (예: +10%%)")
    ap.add_argument("--section", type=int, help="이 번호로 시작하는 절만 측정")
    ap.add_argument("--keep", action="store_true", help="합성한 음성 파일을 남긴다")
    ap.add_argument("--stamp", action="store_true",
                    help="절 제목의 시각 표기를 실측값으로 덮어쓴다")
    ap.add_argument("--demo-sec", type=float, default=180,
                    help="시연 절의 실제 소요 시간(초). 낭독보다 길다")
    args = ap.parse_args()

    ffprobe = find_ffprobe()
    sections = parse(SCRIPT.read_text(encoding="utf-8"))
    if args.section:
        sections = [s for s in sections if s[0].startswith(f"{args.section}.")]
        if not sections:
            raise SystemExit(f"{args.section}번 절을 찾지 못했습니다.")

    tmp = Path(tempfile.mkdtemp(prefix="onfarm_tts_"))
    total, rows = 0.0, []
    for i, (title, body) in enumerate(sections):
        mp3 = tmp / f"{i:02d}.mp3"
        asyncio.run(synth(body, mp3, args.rate))
        sec = duration(mp3, ffprobe)
        total += sec
        rows.append((title, sec, len(body)))

    print(f"음성: {VOICE} · 속도 {args.rate}\n")
    clock = 0.0
    for title, sec, chars in rows:
        print(f"  {int(clock // 60):d}:{int(clock % 60):02d}  {title[:34]:<36} "
              f"{sec:5.1f}초  ({chars}자)")
        clock += sec
    print("─" * 70)
    m, s = divmod(round(total), 60)
    print(f"  낭독 합계 {m}분 {s}초  ({total:.0f}초)")

    if args.section:
        return 0

    demo = next((sec for t, sec, _ in rows if t.startswith("5.")), 0.0)
    slack = TARGET_SEC - total

    if args.stamp:
        # 제목에 적힌 시각이 실측과 어긋나 있으면 리허설이 틀어진다. 실측값으로 덮어쓴다.
        md = SCRIPT.read_text(encoding="utf-8")
        clock = 0.0
        for title, sec, _ in rows:
            base = title.split(" — ")[0]
            stamp = f"{int(clock // 60)}:{int(clock % 60):02d}"
            md = re.sub(
                rf"^## {re.escape(base)}(?: — .*)?$",
                f"## {base} — {stamp}",
                md, count=1, flags=re.MULTILINE,
            )
            clock += args.demo_sec if base.startswith("5.") else sec
        SCRIPT.write_text(md, encoding="utf-8")
        print(f"\n  대본의 시각 표기를 실측값으로 갱신했습니다 "
              f"(시연 {args.demo_sec:.0f}초 가정, 끝 {int(clock // 60)}:{int(clock % 60):02d}).")
    print(f"\n  10분 기준 여유 {slack:+.0f}초")
    print(f"  · 5번 시연 절의 낭독은 {demo:.0f}초지만, 실제 시연은 화면 조작이 섞여 "
          f"보통 낭독의 2배 안팎이 걸린다.")
    print(f"  · 시연을 3분(180초)으로 보면 실제 소요는 약 "
          f"{(total - demo + 180) / 60:.1f}분이다.")
    if total - demo + 180 > TARGET_SEC:
        print("  ⚠ 10분을 넘는다. 절을 줄이거나 --rate +10% 로 읽어야 한다.")

    if args.keep:
        print(f"\n  음성 파일: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
