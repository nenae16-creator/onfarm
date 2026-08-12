"""
시연 영상을 만든다 — 1차 제출의 '시각화 파일' 이자, 발표장에서 서버가 죽었을 때의 대비책.

    python tools/build_demo_video.py --proof   # 장면별 대표 프레임만 뽑아 먼저 확인
    python tools/build_demo_video.py           # 최종 mp4 조립

순서:
  1) 장면별 나레이션을 edge-tts 로 합성하고 길이를 잰다.
  2) 그 길이에 맞춰 화면을 잡아두며 녹화한다(tools/record_demo.mjs).
  3) 자막을 Pillow 로 PNG 렌더해 얹고, 음성과 함께 1920x1080 으로 조립한다.

★ mp4 를 여러 번 다시 뽑지 않는다. --proof 로 장면별 프레임을 먼저 확인하고,
  괜찮을 때 한 번만 조립한다. 영상은 확인 비용이 비싸다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

VOICE = "ko-KR-InJoonNeural"
W, H = 1920, 1080
CREAM, INK, MUTED, ORANGE = "#FDF8F0", "#23201C", "#7A7266", "#DD4B14"
FONT_BOLD = "C:/Windows/Fonts/malgunbd.ttf"
FONT_REG = "C:/Windows/Fonts/malgun.ttf"

# 장면: 화면 동작 + 그 위에 읽을 말. 대본(docs/PITCH_10MIN.md 5절)과 같은 문장을 쓴다.
SCENES = [
    ("home", "천안의 고령 농가가 농산물을 파는 데 걸리는 시간, 3분입니다."),
    ("photo", "농가가 하는 첫 번째 일은 사진을 찍는 것입니다."),
    ("analyze", "사진을 올리면 판정이 시작됩니다."),
    ("candidates", "AI 는 확정하지 않습니다. 물어봅니다. 1번, 2번, 3번 중에 고르시면 됩니다."),
    ("alternative", "후보에 없으면 직접 고를 수 있습니다. AI 가 틀려도 판매는 진행됩니다."),
    ("pick", "번호를 누르면 판매 단위가 나옵니다."),
    ("quantity", "가격은 농가가 정하지 않습니다. 운영자가 미리 정해 둔 표준 가격입니다. "
                 "몇 상자 파실지만 누르시면 됩니다."),
    ("confirm", "이대로 올릴지 확인합니다."),
    ("done", "끝났습니다. 농가가 한 일은 사진 찍기, 번호 고르기, 수량 확인 세 가지뿐입니다."),
    ("store", "글자를 친 곳은 한 곳도 없습니다. 올린 상품은 소비자 화면에 바로 올라갑니다."),
]

PAD_MS = 700  # 말이 끝난 뒤 화면이 잠깐 남는 여유


def tool(name: str) -> str:
    if p := shutil.which(name):
        return p
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    for p in winget.glob(f"Gyan.FFmpeg*/**/bin/{name}.exe"):
        return str(p)
    raise SystemExit(f"{name} 을(를) 찾지 못했습니다. ffmpeg 를 설치하세요.")


FFMPEG, FFPROBE = tool("ffmpeg"), tool("ffprobe")


def duration(path: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def narrate() -> list[dict]:
    """장면별 음성을 만들고 길이를 잰다. 이 길이가 곧 화면을 잡아둘 시간이 된다."""
    import edge_tts

    voices = OUT / "voice"
    voices.mkdir(exist_ok=True)
    scenes = []
    for i, (sid, line) in enumerate(SCENES):
        mp3 = voices / f"{i:02d}_{sid}.mp3"
        asyncio.run(edge_tts.Communicate(line, VOICE).save(str(mp3)))
        sec = duration(mp3)
        scenes.append({"id": sid, "line": line, "mp3": str(mp3),
                       "holdMs": round(sec * 1000) + PAD_MS, "voiceSec": sec})
        print(f"  {sid:<12} {sec:5.1f}초  {line[:34]}")
    return scenes


def caption_png(text: str, path: Path) -> None:
    """자막을 이미지로 만든다 — ffmpeg 문자열 escape 지옥을 피하고 줄바꿈을 직접 통제한다."""
    box_w, box_h = 840, 1080
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_BOLD, 46)
    # 한글은 어절 단위로 끊는다
    lines: list[str] = []
    for para in textwrap.wrap(text, width=17, break_long_words=False):
        lines.append(para)
    total = len(lines) * 66
    y = (box_h - total) // 2
    for ln in lines:
        d.text((0, y), ln, font=font, fill=INK)
        y += 66
    img.save(path)


def record(scenes: list[dict]) -> tuple[Path, list[dict]]:
    (OUT / "scenes.json").write_text(
        json.dumps([{"id": s["id"], "holdMs": s["holdMs"]} for s in scenes],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    node_path = next(
        (str(p.parent) for p in
         (Path.home() / "AppData/Local/npm-cache/_npx").glob("*/node_modules/playwright")),
        "",
    )
    env = {**dict(__import__("os").environ), "NODE_PATH": node_path}
    proc = subprocess.run(
        ["node", "tools/record_demo.mjs", "--scenes", str(OUT / "scenes.json"), "--out", str(OUT)],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"녹화 실패:\n{proc.stderr[-2000:]}")
    raw = Path((OUT / "video_path.txt").read_text(encoding="utf-8").strip())
    timeline = json.loads((OUT / "timeline.json").read_text(encoding="utf-8"))
    return raw, timeline


def proof_frames(video: Path, timeline: list[dict]) -> Path:
    """조립 전에 장면별 대표 프레임을 뽑는다. 여기서 틀린 걸 잡아야 mp4 를 두 번 만들지 않는다."""
    frames = OUT / "proof"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    for i, seg in enumerate(timeline):
        at = seg["start"] + (seg["end"] - seg["start"]) * 0.75  # 동작이 끝난 뒤 지점
        subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(video),
             "-frames:v", "1", str(frames / f"{i:02d}_{seg['id']}.png")],
            check=True,
        )
    return frames


def assemble(video: Path, scenes: list[dict], timeline: list[dict]) -> Path:
    # 녹화 앞머리에는 로그인 처리 시간이 들어 있다. 첫 장면부터 잘라 쓰고,
    # 모든 구간 시각을 그만큼 앞으로 당긴다.
    lead = timeline[0]["start"]
    timeline = [{**t, "start": t["start"] - lead, "end": t["end"] - lead} for t in timeline]

    caps = OUT / "caps"
    caps.mkdir(exist_ok=True)
    for i, s in enumerate(scenes):
        caption_png(s["line"], caps / f"{i:02d}.png")

    # 음성을 장면 시작 시각에 맞춰 배치한다
    concat = OUT / "voice_concat.txt"
    audio = OUT / "narration.m4a"
    inputs, filters, amix = [], [], []
    for i, (s, seg) in enumerate(zip(scenes, timeline)):
        inputs += ["-i", s["mp3"]]
        filters.append(f"[{i}:a]adelay={int(seg['start'] * 1000)}|{int(seg['start'] * 1000)}[a{i}]")
        amix.append(f"[a{i}]")
    total = timeline[-1]["end"]
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", *inputs,
         "-filter_complex", ";".join(filters) + ";" + "".join(amix) +
         f"amix=inputs={len(scenes)}:normalize=0[out]",
         "-map", "[out]", "-t", f"{total:.2f}", str(audio)],
        check=True,
    )
    concat.unlink(missing_ok=True)

    # 배경 + 폰 화면 + 자막
    bg = OUT / "bg.png"
    img = Image.new("RGB", (W, H), CREAM)
    d = ImageDraw.Draw(img)
    d.text((980, 96), "ON-FARM", font=ImageFont.truetype(FONT_BOLD, 58), fill=INK)
    d.text((980, 172), "사진 한 장으로 끝나는 고령농 출하 도구",
           font=ImageFont.truetype(FONT_REG, 30), fill=MUTED)
    d.rectangle([980, 232, 1060, 238], fill=ORANGE)
    # 폰 화면 테두리 — 영상이 배경에 그냥 얹힌 것처럼 보이지 않게 한다.
    # 아래 overlay 가 x=340..774, y=70..1010 을 덮으므로 그보다 조금 크게 그린다.
    d.rounded_rectangle([334, 64, 780, 1016], radius=18, outline="#E1D7C6", width=3)
    d.text((980, 980), "실제 서버 화면을 그대로 녹화했습니다 · 재생 속도 등배",
           font=ImageFont.truetype(FONT_REG, 24), fill=MUTED)
    img.save(bg)

    phone_h = 940
    chain = [
        "[1:v]scale=-2:%d[phone]" % phone_h,
        "[0:v][phone]overlay=x=340:y=(H-h)/2[base0]",
    ]
    last = "base0"
    for i, seg in enumerate(timeline):
        tag = f"base{i + 1}"
        chain.append(
            f"[{last}][{i + 2}:v]overlay=x=980:y=0:"
            f"enable='between(t,{seg['start']:.2f},{seg['end']:.2f})'[{tag}]"
        )
        last = tag

    out = OUT / "ON-FARM_시연.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-loop", "1", "-i", str(bg),
         "-ss", f"{lead:.2f}", "-i", str(video),
         *sum(([("-loop"), "1", "-i", str(caps / f"{i:02d}.png")] for i in range(len(scenes))), []),
         "-i", str(audio),
         "-filter_complex", ";".join(chain),
         "-map", f"[{last}]", "-map", f"{len(scenes) + 2}:a",
         "-t", f"{total:.2f}", "-r", "30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)],
        check=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proof", action="store_true", help="대표 프레임만 뽑고 멈춘다")
    ap.add_argument("--reuse", action="store_true", help="이전 녹화를 다시 쓴다")
    args = ap.parse_args()

    meta = OUT / "narration.json"
    if args.reuse and meta.exists():
        scenes = json.loads(meta.read_text(encoding="utf-8"))
        video = Path((OUT / "video_path.txt").read_text(encoding="utf-8").strip())
        timeline = json.loads((OUT / "timeline.json").read_text(encoding="utf-8"))
        print("이전 녹화를 다시 씁니다.")
    else:
        print("1) 나레이션 합성·길이 측정")
        scenes = narrate()
        meta.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n2) 화면 녹화 (합계 {sum(s['holdMs'] for s in scenes) / 1000:.0f}초)")
        video, timeline = record(scenes)

    if args.proof:
        frames = proof_frames(video, timeline)
        print(f"\n대표 프레임 {len(timeline)}장 → {frames}")
        print("확인한 뒤  python tools/build_demo_video.py --reuse  로 조립하세요.")
        return 0

    print("\n3) 자막·음성 합성 조립")
    mp4 = assemble(video, scenes, timeline)
    print(f"\n✔ {mp4.relative_to(ROOT)}  "
          f"({mp4.stat().st_size / 1024 / 1024:.1f} MB · {duration(mp4):.0f}초)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
