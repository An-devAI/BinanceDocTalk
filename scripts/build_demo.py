"""Capture REAL offline CLI outputs and create a clearly labelled recorded walkthrough.

Pillow is used only for presentation frames, never to fabricate predictions or quality scores.
Optional GStreamer can convert the MJPEG AVI into a broadly playable WebM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from io import BytesIO
import json
from pathlib import Path
import struct
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    "How is a market order different from a limit order?",
    "A stop-limit order guarantees execution after triggering, right?",
    "What is the exact Spot fee for my account today?",
]


def chunk(tag: bytes, data: bytes) -> bytes:
    return tag + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def avi(frames: list[bytes], width: int, height: int) -> bytes:
    maximum = max(map(len, frames))
    avih = struct.pack("<14I", 1_000_000, maximum, 0, 0x10, len(frames), 0, 1,
                       maximum, width, height, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHH8I4h", b"vids", b"MJPG", 0, 0, 0, 0, 1, 1, 0,
                       len(frames), maximum, 0xffffffff, 0, 0, 0, width, height)
    strf = struct.pack("<IiiHH4sIiiII", 40, width, height, 1, 24, b"MJPG", width*height*3,
                       0, 0, 0, 0)
    header = chunk(b"LIST", b"hdrl" + chunk(b"avih", avih)
                   + chunk(b"LIST", b"strl" + chunk(b"strh", strh) + chunk(b"strf", strf)))
    movie, index, offset = [], [], 4
    for frame in frames:
        frame_chunk = chunk(b"00dc", frame)
        movie.append(frame_chunk)
        index.append(struct.pack("<4sIII", b"00dc", 0x10, offset, len(frame)))
        offset += len(frame_chunk)
    return chunk(b"RIFF", b"AVI " + header + chunk(b"LIST", b"movi" + b"".join(movie))
                 + chunk(b"idx1", b"".join(index)))


def main():
    output = ROOT / "demo"
    output.mkdir(exist_ok=True)
    captures = []
    for question in QUESTIONS:
        process = subprocess.run([sys.executable, "scripts/doctalk.py", "demo", "--question", question],
                                 cwd=ROOT, capture_output=True, text=True, check=True)
        captures.append(json.loads(process.stdout))
    record = {"recorded_at": datetime.now(timezone.utc).isoformat(), "mode": "demo_extract",
              "description": "Actual offline CLI outputs on original synthetic notes; not LLM quality results",
              "cases": captures}
    (output / "recorded_outputs.json").write_text(json.dumps(record, indent=2) + "\n")
    ui = (ROOT / "src/binance_doctalk/ui.html").read_text()
    ui = ui.replace("INDEPENDENT EDUCATIONAL PROJECT · v0.1", "RECORDED CLI OUTPUTS · OFFLINE WALKTHROUGH")
    ui = ui.replace('<select id="mode"', '<select disabled id="mode"')
    ui = ui.replace('<textarea id="question"', '<textarea readonly id="question"')
    ui = ui.replace('</textarea>', escape(QUESTIONS[0]) + '</textarea>')
    ui = ui.replace('Single-turn questions · Educational use only', '3 recorded cases · Run the CLI for new questions')
    ui = ui.replace('Offline preview uses original educational notes and deterministic retrieval. No LLM, no official Binance corpus, and no API charges.',
                    'Recorded offline CLI outputs. Select a captured example below, then inspect its original notes. No live inference or official-source quality measurement.')
    data = json.dumps({r["question"]: r for r in captures}).replace("<", "\\u003c")
    status = json.dumps({"api_key_configured": False, "live_index_loaded": False,
                         "demo": {"document_count": 12}, "quality_results": "not_measured",
                         "benchmark": {"questions": 80, "human_reviewed": 0}})
    fixture = f'''<script>
const recordedCases={data};
window.fetch=async (path,options)=>{{
 if(path==='/api/status')return {{json:async()=>({status})}};
 const q=JSON.parse(options.body).question;
 if(!recordedCases[q])throw new Error('Only recorded examples are available');
 const result=JSON.parse(JSON.stringify(recordedCases[q]));
 const notes={json.dumps({h['chunk_id']: next(c['text'] for c in r['claims'] if h['chunk_id'] in c['evidence_ids']) for r in captures for h in r['retrieved'] if any(h['chunk_id'] in c['evidence_ids'] for c in r['claims'])}).replace('<', chr(92)+'u003c')};
 result.retrieved=result.retrieved.filter(h=>notes[h.chunk_id]).map(h=>({{...h,text:notes[h.chunk_id]}}));
 return {{json:async()=>result}};
}};
</script>'''
    ui = ui.replace('<script>\nconst $=', fixture + '\n<script>\nconst $=')
    (output / "recorded-preview.html").write_text(ui)
    from PIL import Image, ImageDraw, ImageFont
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    regular = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 18)
    title = ImageFont.truetype(font_path, 35)
    frames = []
    for i, case in enumerate(captures):
        image = Image.new("RGB", (1280, 720), "#14241e")
        draw = ImageDraw.Draw(image)
        draw.text((55, 35), "BinanceDocTalk", fill="#ece9c7", font=title)
        draw.text((55, 95), "RECORDED OFFLINE CLI OUTPUTS | NO LLM | $0 API", fill="#afd5b5", font=small)
        draw.rounded_rectangle((40, 140, 1240, 650), radius=16, fill="#1e342b")
        y = 166
        draw.text((65, y), f"CASE {i+1}/3 · {case['status'].upper()}", fill="#e8d990", font=regular)
        y += 52
        for line in textwrap.wrap("$ " + case["question"], 78):
            draw.text((65, y), line, fill="#ffffff", font=regular)
            y += 32
        y += 28
        content = " ".join(c["text"] for c in case["claims"]) or case["message"]
        for line in textwrap.wrap(content, 82):
            draw.text((65, y), line, fill="#d7e3d8", font=regular)
            y += 32
        if case["claims"]:
            y += 24
            ids = " · ".join(cid for c in case["claims"] for cid in c["evidence_ids"])
            draw.text((65, y), "Original note: " + ids, fill="#e8d990", font=small)
        draw.text((55, 680), "Not official Binance content · Human gold review and held-out LLM results are pending", fill="#b0c3b3", font=small)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        frames.extend([buffer.getvalue()] * 9)
        if i == 0:
            image.save(output / "cli-preview.png")
    local = ROOT / "artifacts/local"
    local.mkdir(parents=True, exist_ok=True)
    (local / "offline-walkthrough.avi").write_bytes(avi(frames, 1280, 720))
    print("Captured 3 real CLI outputs; wrote standalone recorded preview, poster, and 27-second AVI.")


if __name__ == "__main__":
    main()
