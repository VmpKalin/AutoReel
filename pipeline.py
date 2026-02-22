#!/usr/bin/env python3
"""
Video Processing Pipeline
=========================
1. Extract audio
2. Enhance audio (Auphonic)
3. Merge enhanced audio into video
4. Transcription (WhisperX) → save locally
5. Text correction (Ollama)
6. Subtitle generation (SRT + ASS)
7. Burn subtitles into video
8. Video formatting (padding, crop)
9. Metadata generation for Instagram + TikTok
"""

import os
import re
import json
import time
import shutil
import logging
import argparse
import subprocess
import requests
import ollama
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
import pysrt

load_dotenv()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
class Config:
    # Auphonic
    AUPHONIC_API_KEY = os.getenv("AUPHONIC_API_KEY", "")
    AUPHONIC_PRESET  = os.getenv("AUPHONIC_PRESET", "")


    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

    # WhisperX
    WHISPER_MODEL    = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_DEVICE   = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

    # Subtitles
    SUBTITLE_FONT_PATH     = os.getenv("SUBTITLE_FONT_PATH", "/System/Library/Fonts/Supplemental/Impact.ttf")
    SUBTITLE_FONT          = os.getenv("SUBTITLE_FONT", "Impact")
    SUBTITLE_FONT_SIZE     = int(os.getenv("SUBTITLE_FONT_SIZE", "18"))
    SUBTITLE_COLOR         = os.getenv("SUBTITLE_COLOR", "&H00FFFFFF")
    SUBTITLE_OUTLINE_COLOR = os.getenv("SUBTITLE_OUTLINE_COLOR", "&H00000000")
    SUBTITLE_OUTLINE_SIZE  = int(os.getenv("SUBTITLE_OUTLINE_SIZE", "2"))
    SUBTITLE_POSITION      = os.getenv("SUBTITLE_POSITION", "bottom")
    SUBTITLE_BOTTOM_MARGIN = int(os.getenv("SUBTITLE_BOTTOM_MARGIN", "80"))

    # Video format
    OUTPUT_FORMAT    = os.getenv("OUTPUT_FORMAT", "9:16")
    ADD_PADDING      = os.getenv("ADD_PADDING", "false").lower() == "true"
    PADDING_COLOR    = os.getenv("PADDING_COLOR", "black")
    CONVERT_TO_1080P = os.getenv("CONVERT_TO_1080P", "true").lower() == "true"

    VIDEO_SPEED = float(os.getenv("VIDEO_SPEED", "1.0"))  # 1.0 = normal, 1.5 = 1.5x, 2.0 = 2x

cfg = Config()


# ─────────────────────────────────────────────
# AI Prompts
# ─────────────────────────────────────────────
TRANSCRIPT_FIX_PROMPT = """You are a professional transcript editor.
Your task is to fix grammar, punctuation and spelling errors in each line.

Rules:
- KEEP the [N] numbering and structure exactly as is
- DO NOT change the meaning, word order or content
- Fix punctuation, capitalization and obvious speech-to-text errors
- Do NOT add periods at the end of lines — subtitles should not end with a dot
- Return ONLY the corrected lines in format [N] text
- Nothing else, no explanations"""


METADATA_PROMPT = """You are a social media content strategist specializing in Instagram and TikTok growth.
Based on the video transcript, generate optimized metadata for maximum reach and engagement.

Return ONLY valid JSON, no markdown, no extra text:
{
  "title": "Catchy video title under 60 chars, curiosity-driven",
  "instagram_caption": "Engaging caption with hook, value, CTA and relevant hashtags. Max 2200 chars. Use emojis naturally.",
  "instagram_hashtags": ["hashtag1", "hashtag2"],
  "tiktok_caption": "Short punchy caption under 150 chars with 3-5 hashtags. Hook in first line.",
  "tiktok_hashtags": ["hashtag1", "hashtag2"],
  "short_summary": "2 sentences describing the video content"
}

Hashtag strategy:
- Instagram: mix of niche (10k-100k), medium (100k-1M) and broad tags
- TikTok: trending + niche specific tags
- Always include topic-relevant tags, avoid generic spam tags"""


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def run(cmd: str, desc: str = "") -> str:
    log.info(f"⚙️  {desc or cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_srt_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_ms = int(td.total_seconds() * 1000)
    h  = total_ms // 3_600_000
    m  = (total_ms % 3_600_000) // 60_000
    s  = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# ─────────────────────────────────────────────
# Step 1 — Extract Audio
# ─────────────────────────────────────────────
def extract_audio(video_path: Path, output_dir: Path) -> Path:
    audio_path = output_dir / "audio_original.wav"
    run(
        f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 1 "{audio_path}"',
        "Extract audio from video",
    )
    log.info(f"✅ Audio saved: {audio_path}")
    return audio_path


# ─────────────────────────────────────────────
# Step 2 — Enhance Audio via Auphonic
# ─────────────────────────────────────────────
def enhance_audio_auphonic(audio_path: Path, output_dir: Path) -> Path:
    if not cfg.AUPHONIC_API_KEY:
        log.warning("⚠️  AUPHONIC_API_KEY is not set, skipping audio enhancement")
        return audio_path

    log.info("🔊 Uploading audio to Auphonic...")
    headers = {"Authorization": f"Bearer {cfg.AUPHONIC_API_KEY}"}

    data = {
        "action": "start",
        "output_basename": "enhanced",
        "algorithms": json.dumps({
            "normloudness": True,
            "denoise": True,
            "denoiseamount": 0.8,
            "hiss_reduction": True,
        }),
    }
    if cfg.AUPHONIC_PRESET:
        data["preset"] = cfg.AUPHONIC_PRESET

    with open(audio_path, "rb") as f:
        resp = requests.post(
            "https://auphonic.com/api/simple/productions.json",
            headers=headers,
            data=data,
            files={"input_file": f},
        )
    resp.raise_for_status()
    uuid = resp.json()["data"]["uuid"]
    log.info(f"📤 Production UUID: {uuid}")

    log.info("⏳ Waiting for Auphonic to finish...")
    prod_data = {}
    for _ in range(120):
        time.sleep(5)
        status_resp = requests.get(
            f"https://auphonic.com/api/production/{uuid}.json", headers=headers
        )
        status_resp.raise_for_status()
        prod_data = status_resp.json()["data"]
        log.info(f"   Status: {prod_data.get('status_string', '')}")
        if prod_data["status"] == 3:
            break
        if prod_data["status"] in (9, 10):
            raise RuntimeError(f"Auphonic failed: {prod_data.get('status_string')}")

    output_files = prod_data.get("output_files", [])
    if not output_files:
        raise RuntimeError("Auphonic did not return output files")

    enhanced_path = output_dir / "audio_enhanced.wav"
    log.info("📥 Downloading enhanced audio...")
    audio_resp = requests.get(output_files[0]["download_url"], headers=headers)
    audio_resp.raise_for_status()
    with open(enhanced_path, "wb") as f:
        f.write(audio_resp.content)

    log.info(f"✅ Enhanced audio saved: {enhanced_path}")
    return enhanced_path


# ─────────────────────────────────────────────
# Step 3 — Merge Enhanced Audio into Video
# ─────────────────────────────────────────────
def merge_audio_video(video_path: Path, audio_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / "video_enhanced.mp4"
    run(
        f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
        f'-c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}"',
        "Merge video with enhanced audio",
    )
    log.info(f"✅ Video with new audio: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Step 4 — Transcription (WhisperX) → save locally
# ─────────────────────────────────────────────
def transcribe(audio_path: Path, output_dir: Path) -> list[dict]:
    try:
        import whisperx
    except ImportError:
        raise ImportError("WhisperX is not installed. Run: pip install whisperx torch")

    device = cfg.WHISPER_DEVICE
    log.info(f"🎙️  WhisperX transcription (model={cfg.WHISPER_MODEL}, device={device})...")

    model  = whisperx.load_model(cfg.WHISPER_MODEL, device=device, compute_type="float32")
    audio  = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=cfg.WHISPER_LANGUAGE, batch_size=16)

    log.info("🔡 Word-level alignment...")
    model_a, metadata = whisperx.load_align_model(
        language_code=cfg.WHISPER_LANGUAGE, device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
    )

    word_segments = result.get("word_segments", [])
    segments = []

    if word_segments:
        log.info(f"✂️  Building subtitles from {len(word_segments)} words...")
        MAX_CHARS    = 42
        MAX_DURATION = 3.5

        chunk_words = []
        chunk_start = None  # буде встановлено з реального початку першого слова

        for word in word_segments:
            if "start" not in word or "end" not in word:
                chunk_words.append(word["word"])
                continue

            # Встановлюємо start з реального початку першого слова в chunk
            if chunk_start is None:
                chunk_start = word["start"]

            chunk_words.append(word["word"])
            chunk_text = " ".join(chunk_words)
            chunk_dur  = word["end"] - chunk_start

            if len(chunk_text) >= MAX_CHARS or chunk_dur >= MAX_DURATION:
                segments.append({
                    "start": chunk_start,
                    "end":   word["end"],
                    "text":  chunk_text,
                })
                chunk_words = []
                chunk_start = None  # скидаємо, наступне слово встановить новий start

        if chunk_words and chunk_start is not None:
            last_end = word_segments[-1].get("end", chunk_start + 1)
            segments.append({
                "start": chunk_start,
                "end":   last_end,
                "text":  " ".join(chunk_words),
            })
    else:
        segments = result["segments"]

    log.info(f"✅ Transcription: {len(segments)} subtitle segments")

    raw_json = output_dir / "transcript_raw.json"
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    raw_txt = output_dir / "transcript.txt"
    with open(raw_txt, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{format_srt_time(seg['start'])} → {format_srt_time(seg['end'])}] {seg['text'].strip()}\n")

    log.info(f"💾 Transcript: {raw_json}")
    return segments


# ─────────────────────────────────────────────
# Step 5 — Fix Transcript via Ollama
# ─────────────────────────────────────────────
def fix_transcript(segments: list[dict], output_dir: Path) -> list[dict]:
    log.info("✍️  Fixing text with local model...")
    full_text = "\n".join(f"[{i}] {seg['text']}" for i, seg in enumerate(segments))

    message = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": TRANSCRIPT_FIX_PROMPT},
            {"role": "user", "content": full_text},
        ]
    )

    corrected_lines = {}
    response_text = message['message']['content']
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("["):
            try:
                idx_end = line.index("]")
                idx  = int(line[1:idx_end])
                text = line[idx_end + 1:].strip()
                corrected_lines[idx] = text
            except (ValueError, IndexError):
                continue

    for i, seg in enumerate(segments):
        if i in corrected_lines:
            seg["text"] = corrected_lines[i]

    fixed_json = output_dir / "transcript_fixed.json"
    with open(fixed_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    fixed_txt = output_dir / "transcript_fixed.txt"
    with open(fixed_txt, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{format_srt_time(seg['start'])} → {format_srt_time(seg['end'])}] {seg['text'].strip()}\n")

    log.info(f"✅ Corrected transcript: {fixed_json}")
    return segments


def remove_subtitle_duplicates(segments: list[dict]) -> list[dict]:
    """Removes duplicated text between adjacent segments."""
    if len(segments) < 2:
        return segments

    for i in range(1, len(segments)):
        prev_text  = segments[i - 1]["text"].strip().lower()
        curr_text  = segments[i]["text"].strip()
        curr_lower = curr_text.lower()

        words_prev = prev_text.split()
        words_curr = curr_lower.split()

        overlap = 0
        for size in range(min(8, len(words_prev), len(words_curr)), 0, -1):
            if words_prev[-size:] == words_curr[:size]:
                overlap = size
                break

        if overlap > 0:
            original_words = curr_text.split()
            segments[i]["text"] = " ".join(original_words[overlap:]).strip()
            log.info(f"🧹 Removed duplicate in segment {i}: {overlap} words")

    segments = [s for s in segments if s.get("text", "").strip()]
    return segments


# ─────────────────────────────────────────────
# Step 6 — Generate Subtitles (SRT + ASS)
# ─────────────────────────────────────────────
def generate_srt(segments: list[dict], output_dir: Path) -> Path:
    srt_path = output_dir / "subtitles.srt"
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(
            f"{i}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n{seg['text'].strip()}\n"
        )
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"✅ SRT subtitles: {srt_path}")
    return srt_path


def generate_ass(segments: list[dict], output_dir: Path) -> Path:
    ass_path = output_dir / "subtitles.ass"
    alignment = {"bottom": 2, "top": 8, "center": 5}.get(cfg.SUBTITLE_POSITION, 2)

    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        "PlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{cfg.SUBTITLE_FONT},{cfg.SUBTITLE_FONT_SIZE},"
        f"{cfg.SUBTITLE_COLOR},&H000000FF,{cfg.SUBTITLE_OUTLINE_COLOR},&H00000000,"
        f"0,0,0,0,100,100,0,0,1,{cfg.SUBTITLE_OUTLINE_SIZE},0,{alignment},10,10,30,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [
        f"Dialogue: 0,{format_ass_time(s['start'])},{format_ass_time(s['end'])},"
        f"Default,,0,0,0,,{s['text'].strip().replace(chr(10), chr(92)+'N')}"
        for s in segments
    ]
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))
    log.info(f"✅ ASS subtitles: {ass_path}")
    return ass_path


# ─────────────────────────────────────────────
# Step 7 — Burn Subtitles into Video
# ─────────────────────────────────────────────
def burn_subtitles(video_path: Path, srt_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / "video_subtitled.mp4"
    log.info("🎬 Burning subtitles into video...")

    working_path = output_dir / "video_h264.mp4"
    if cfg.CONVERT_TO_1080P:
        log.info("🔄 Converting to h264 1080p...")
        subprocess.run(
            f'ffmpeg -y -i "{video_path}" -c:v libx264 -crf 18 -preset fast '
            f'-vf "scale=1080:1920,format=yuv420p" -c:a aac "{working_path}"',
            shell=True, capture_output=True
        )
        log.info("✅ Conversion done")
        video = VideoFileClip(str(working_path))

    subs = pysrt.open(str(srt_path))
    auto_font_size = max(cfg.SUBTITLE_FONT_SIZE, int(video.h * 0.025))

    black_bg = ColorClip(
        size=(video.w, video.h),
        color=[0, 0, 0],
        duration=video.duration
    ).with_fps(video.fps)

    subtitle_clips = []
    for sub in subs:
        start    = sub.start.ordinal / 1000.0
        end      = sub.end.ordinal / 1000.0
        duration = end - start
        if duration <= 0:
            continue

        txt_clip = (
            TextClip(
                text=sub.text.strip() + "\n",  # extra line for descenders (g, p, y, ,)
                font=cfg.SUBTITLE_FONT_PATH,
                font_size=auto_font_size,
                color="white",
                stroke_color="black",
                stroke_width=max(cfg.SUBTITLE_OUTLINE_SIZE, 3),
                method="caption",
                size=(int(video.w * 0.85), None),
            )
            .with_start(start)
            .with_duration(duration)
        )

        if cfg.SUBTITLE_POSITION == "top":
            pos = ("center", cfg.SUBTITLE_BOTTOM_MARGIN)
        elif cfg.SUBTITLE_POSITION == "center":
            pos = ("center", "center")
        else:  # bottom
            DESCENDER_BUFFER = 20
            text_height_with_buffer = txt_clip.h + max(cfg.SUBTITLE_OUTLINE_SIZE, 3) * 4 + DESCENDER_BUFFER
            bottom_y = video.h - text_height_with_buffer - cfg.SUBTITLE_BOTTOM_MARGIN
            bottom_y = max(bottom_y, cfg.SUBTITLE_BOTTOM_MARGIN)
            pos = ("center", bottom_y)

        subtitle_clips.append(txt_clip.with_position(pos))

    subtitle_layer_path = output_dir / "subtitle_layer.mp4"
    subtitle_layer = CompositeVideoClip([black_bg, *subtitle_clips])
    subtitle_layer.write_videofile(
        str(subtitle_layer_path),
        fps=video.fps,
        codec="libx264",
        audio=False,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
        logger=None,
    )
    video.close()

    log.info("🎨 Overlaying subtitles via ffmpeg (preserving colors)...")
    result = subprocess.run(
        f'ffmpeg -y -i "{working_path}" -i "{subtitle_layer_path}" '
        f'-filter_complex "[1:v]colorkey=0x000000:0.15:0.1[sub];[0:v][sub]overlay" '
        f'-c:v libx264 -crf 18 -preset fast -c:a copy "{output_path}"',
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error(f"ffmpeg overlay error: {result.stderr}")
        raise RuntimeError("Failed to overlay subtitles")

    subtitle_layer_path.unlink(missing_ok=True)
    log.info(f"✅ Video with subtitles: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Step 8 — Format Video (padding / crop)
# ─────────────────────────────────────────────
ASPECT_RATIOS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1":  (1080, 1080),
    "4:5":  (1080, 1350),
}


def format_video(video_path: Path, output_dir: Path, suffix: str = "formatted") -> Path:
    if cfg.OUTPUT_FORMAT == "original" and not cfg.ADD_PADDING:
        log.info("⏩ Formatting skipped (original)")
        return video_path

    output_path = output_dir / f"video_{suffix}.mp4"
    w, h = ASPECT_RATIOS.get(cfg.OUTPUT_FORMAT, (1920, 1080))

    if cfg.ADD_PADDING:
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:{cfg.PADDING_COLOR}"
        )
    else:
        vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"

    run(
        f'ffmpeg -y -i "{video_path}" -vf "{vf}" -c:a copy "{output_path}"',
        f"Video formatting → {cfg.OUTPUT_FORMAT}",
    )
    log.info(f"✅ Formatted video: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Step 9 — Generate Metadata (Instagram + TikTok)
# ─────────────────────────────────────────────
def generate_metadata(segments: list[dict], output_dir: Path) -> dict:
    log.info("📋 Generating metadata (Instagram + TikTok) via Ollama...")

    transcript_preview = " ".join(seg["text"] for seg in segments)[:3000]

    message = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": METADATA_PROMPT},
            {"role": "user", "content": f"Video transcript:\n{transcript_preview}"},
        ]
    )

    raw = message['message']['content'].strip()

    # Витягуємо JSON навіть якщо є текст навколо
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"⚠️  Failed to parse metadata as JSON: {e}")
        metadata = {"raw": raw}

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log.info(f"✅ Metadata saved: {metadata_path}")
    return metadata


def print_metadata(metadata: dict):
    if not metadata:
        return
    print("\n" + "=" * 55)
    print("📋 GENERATED METADATA")
    print("=" * 55)
    list_fields = {"instagram_hashtags", "tiktok_hashtags"}
    for k, v in metadata.items():
        if k in list_fields:
            print(f"\n🔹 {k.upper()}:\n   {' '.join('#' + t for t in v)}")
        else:
            print(f"\n🔹 {k.upper()}:\n{v}")
    print()

def send_metadata_to_telegram(metadata: dict, video_name: str):
    token   = cfg.TELEGRAM_BOT_TOKEN
    chat_id = cfg.TELEGRAM_CHAT_ID

    if not token or not chat_id:
        log.warning("⚠️  TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping")
        return

    hashtags_ig  = " ".join(f"#{t}" for t in metadata.get("instagram_hashtags", []))
    hashtags_tt  = " ".join(f"#{t}" for t in metadata.get("tiktok_hashtags", []))

    text = (
        f"🎬 *{video_name}*\n\n"
        f"📌 *Title:* {metadata.get('title', '')}\n\n"
        f"📝 *Summary:* {metadata.get('short_summary', '')}\n\n"
        f"📸 *Instagram caption:*\n{metadata.get('instagram_caption', '')}\n\n"
        f"🏷 *Instagram hashtags:*\n{hashtags_ig}\n\n"
        f"🎵 *TikTok caption:*\n{metadata.get('tiktok_caption', '')}\n\n"
        f"🏷 *TikTok hashtags:*\n{hashtags_tt}"
    )

    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
    )

    if resp.status_code == 200:
        log.info("✅ Metadata sent to Telegram")
    else:
        log.warning(f"⚠️  Telegram error: {resp.text}")

def speed_up_video(video_path: Path, output_dir: Path) -> Path:
    if cfg.VIDEO_SPEED == 1.0:
        log.info("⏩ Speed change skipped (VIDEO_SPEED=1.0)")
        return video_path

    output_path = output_dir / f"video_speed_{cfg.VIDEO_SPEED}x.mp4"
    log.info(f"⚡ Changing video speed to {cfg.VIDEO_SPEED}x...")

    # setpts для відео, atempo для аудіо (підтримує 0.5-2.0)
    pts = 1.0 / cfg.VIDEO_SPEED
    
    # atempo підтримує тільки 0.5-2.0, для більших значень каскадуємо
    speed = cfg.VIDEO_SPEED
    atempo_filters = []
    while speed > 2.0:
        atempo_filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        atempo_filters.append("atempo=0.5")
        speed /= 0.5
    atempo_filters.append(f"atempo={speed:.4f}")
    atempo = ",".join(atempo_filters)

    run(
        f'ffmpeg -y -i "{video_path}" '
        f'-filter_complex "[0:v]setpts={pts:.4f}*PTS[v];[0:a]{atempo}[a]" '
        f'-map "[v]" -map "[a]" '
        f'-c:v libx264 -crf 18 -preset fast "{output_path}"',
        f"Speed change → {cfg.VIDEO_SPEED}x",
    )

    log.info(f"✅ Speed changed: {output_path}")
    return output_path

# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def run_pipeline(input_video: str, output_dir: str, steps: list[str] = None):
    video_path = Path(input_video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out = ensure_dir(Path(output_dir))
    log.info(f"🚀 Starting pipeline for: {video_path.name}")
    log.info(f"📁 Output directory: {out}")

    all_steps = ["audio", "enhance", "merge", "transcribe", "fix", "subtitles", "format", "speed", "metadata"]
    active = set(steps) if steps else set(all_steps)
    state  = {"video": video_path}

    # 1. Extract audio
    if "audio" in active:
        state["audio"] = extract_audio(state["video"], out)

    # 2. Enhance audio
    if "enhance" in active and "audio" in state:
        state["enhanced_audio"] = enhance_audio_auphonic(state["audio"], out)
    else:
        state["enhanced_audio"] = state.get("audio", video_path)

    # 3. Merge video + audio
    if "merge" in active and state.get("enhanced_audio") != state.get("audio"):
        state["video"] = merge_audio_video(state["video"], state["enhanced_audio"], out)
    else:
        video_subtitled = out / "video_subtitled.mp4"
        video_enhanced  = out / "video_enhanced.mp4"

        if video_subtitled.exists() and "subtitles" not in active:
            state["video"] = video_subtitled
            log.info(f"📂 Using existing subtitled video: {video_subtitled}")
        elif video_enhanced.exists():
            state["video"] = video_enhanced
            log.info(f"📂 Using enhanced video: {video_enhanced}")
        else:
            dest = out / video_path.name
            if not dest.exists():
                shutil.copy2(video_path, dest)
            state["video"] = dest
            log.info(f"📂 Using original video: {dest}")

    # Load existing transcript if present
    fixed_json = out / "transcript_fixed.json"
    raw_json   = out / "transcript_raw.json"
    if "transcribe" not in active:
        if fixed_json.exists():
            with open(fixed_json, encoding="utf-8") as f:
                state["segments"] = json.load(f)
            log.info(f"📂 Loaded existing transcript: {fixed_json}")
        elif raw_json.exists():
            with open(raw_json, encoding="utf-8") as f:
                state["segments"] = json.load(f)
            log.info(f"📂 Loaded existing transcript: {raw_json}")

    # 4. Transcription
    if "transcribe" in active:
        audio_src = state.get("enhanced_audio") or state.get("audio")
        state["segments"] = transcribe(audio_src, out)

    # 5. Text correction
    if "fix" in active and "segments" in state:
        state["segments"] = fix_transcript(state["segments"], out)
        state["segments"] = remove_subtitle_duplicates(state["segments"])

    # 6. Subtitles
    if "subtitles" in active and "segments" in state:
        srt_path = generate_srt(state["segments"], out)
        generate_ass(state["segments"], out)
        state["video"] = burn_subtitles(state["video"], srt_path, out)

    # 7. Video formatting
    if "format" in active:
        state["video"] = format_video(state["video"], out)

    # 8. Speed
    if "speed" in active:
        state["video"] = speed_up_video(state["video"], out)

    # 9. Metadata
    if "metadata" in active and "segments" in state:
        state["metadata"] = generate_metadata(state["segments"], out)
        print_metadata(state["metadata"])
        send_metadata_to_telegram(state["metadata"], video_path.name)

    log.info("\n" + "=" * 55)
    log.info("🎉 PIPELINE COMPLETED!")
    log.info(f"📹 Final video : {state['video']}")
    log.info(f"📁 All files   : {out}")
    return state


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Video pipeline: audio → transcription → subtitles → metadata"
    )
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("-o", "--output", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument(
        "--steps", nargs="+",
        choices=["audio", "enhance", "merge", "transcribe", "fix", "subtitles", "format", "speed", "metadata"],
        help="Run only specified steps (default: all)",
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.steps)


if __name__ == "__main__":
    main()