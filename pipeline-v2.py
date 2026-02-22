#!/usr/bin/env python3
"""
Video Processing Pipeline
=========================
1. Витяг аудіо
2. Покращення звуку (Auphonic)
3. Транскрипція (WhisperX) → зберігаємо локально
4. Виправлення тексту (Claude API)
5. Генерація субтитрів (SRT + ASS)
6. Накладання субтитрів на відео
7. Форматування відео (рамка, crop)
8. Генерація метаданих для Instagram + TikTok
"""

import os
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
import anthropic
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
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

    # Anthropic
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # WhisperX
    WHISPER_MODEL    = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_DEVICE   = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "uk")

    # Subtitles
    SUBTITLE_FONT          = os.getenv("SUBTITLE_FONT", "Arial")
    SUBTITLE_FONT_SIZE     = int(os.getenv("SUBTITLE_FONT_SIZE", "18"))
    SUBTITLE_COLOR         = os.getenv("SUBTITLE_COLOR", "&H00FFFFFF")
    SUBTITLE_OUTLINE_COLOR = os.getenv("SUBTITLE_OUTLINE_COLOR", "&H00000000")
    SUBTITLE_OUTLINE_SIZE  = int(os.getenv("SUBTITLE_OUTLINE_SIZE", "2"))
    SUBTITLE_POSITION      = os.getenv("SUBTITLE_POSITION", "bottom")

    # Video format
    OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "16:9")
    ADD_PADDING   = os.getenv("ADD_PADDING", "false").lower() == "true"
    PADDING_COLOR = os.getenv("PADDING_COLOR", "black")
    
    CONVERT_TO_1080P = os.getenv("CONVERT_TO_1080P", "true").lower() == "true"

    REMOVE_AUPHONIC_WATERMARK = os.getenv("REMOVE_AUPHONIC_WATERMARK", "true").lower() == "true"

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

Return ONLY JSON, no markdown:
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
        "Витяг аудіо з відео",
    )
    log.info(f"✅ Аудіо збережено: {audio_path}")
    return audio_path


# ─────────────────────────────────────────────
# Step 2 — Enhance Audio via Auphonic
# ─────────────────────────────────────────────
def enhance_audio_auphonic(audio_path: Path, output_dir: Path) -> Path:
    if not cfg.AUPHONIC_API_KEY:
        log.warning("⚠️  AUPHONIC_API_KEY не задано, пропускаємо покращення звуку")
        return audio_path

    log.info("🔊 Завантаження аудіо в Auphonic...")
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

    # Polling до завершення
    log.info("⏳ Чекаємо завершення Auphonic...")
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
        raise RuntimeError("Auphonic не повернув файли результату")

    enhanced_path = output_dir / "audio_enhanced.wav"
    log.info("📥 Завантаження покращеного аудіо...")
    audio_resp = requests.get(output_files[0]["download_url"], headers=headers)
    audio_resp.raise_for_status()
    with open(enhanced_path, "wb") as f:
        f.write(audio_resp.content)

    log.info(f"✅ Покращене аудіо збережено: {enhanced_path}")
    return enhanced_path


# ─────────────────────────────────────────────
# Step 3 — Merge Enhanced Audio into Video
# ─────────────────────────────────────────────
def merge_audio_video(video_path: Path, audio_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / "video_enhanced.mp4"
    run(
        f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
        f'-c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}"',
        "Злиття відео з покращеним аудіо",
    )
    log.info(f"✅ Відео з новим звуком: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Step 4 — Transcription (WhisperX) → save locally
# ─────────────────────────────────────────────
def transcribe(audio_path: Path, output_dir: Path) -> list[dict]:
    try:
        import whisperx
    except ImportError:
        raise ImportError("WhisperX не встановлено. Виконай: pip install whisperx torch")

    device = cfg.WHISPER_DEVICE
    log.info(f"🎙️  WhisperX транскрипція (model={cfg.WHISPER_MODEL}, device={device})...")

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

    # Будуємо короткі сегменти прямо з word_segments
    word_segments = result.get("word_segments", [])
    segments = []

    if word_segments:
        log.info(f"✂️  Будуємо субтитри з {len(word_segments)} слів...")
        MAX_CHARS    = 42
        MAX_DURATION = 3.5

        chunk_words  = []
        chunk_start  = word_segments[0]["start"]

        for word in word_segments:
            # Пропускаємо слова без тайм-кодів
            if "start" not in word or "end" not in word:
                chunk_words.append(word["word"])
                continue

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
                chunk_start = word["end"]

        if chunk_words:
            last_end = word_segments[-1].get("end", chunk_start + 1)
            segments.append({
                "start": chunk_start,
                "end":   last_end,
                "text":  " ".join(chunk_words),
            })
    else:
        # Fallback — використовуємо звичайні segments
        segments = result["segments"]

    log.info(f"✅ Транскрипція: {len(segments)} субтитр-сегментів")

    # Зберігаємо
    raw_json = output_dir / "transcript_raw.json"
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    raw_txt = output_dir / "transcript.txt"
    with open(raw_txt, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{format_srt_time(seg['start'])} → {format_srt_time(seg['end'])}] {seg['text'].strip()}\n")

    log.info(f"💾 Транскрипт: {raw_json}")
    return segments


# ─────────────────────────────────────────────
# Step 5 — Fix Transcript via Claude
# ─────────────────────────────────────────────
def fix_transcript(segments: list[dict], output_dir: Path) -> list[dict]:

    log.info("✍️  Виправлення тексту через локальну модель...")
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

    # Зберігаємо виправлений варіант
    fixed_json = output_dir / "transcript_fixed.json"
    with open(fixed_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    fixed_txt = output_dir / "transcript_fixed.txt"
    with open(fixed_txt, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{format_srt_time(seg['start'])} → {format_srt_time(seg['end'])}] {seg['text'].strip()}\n")

    log.info(f"✅ Виправлений транскрипт: {fixed_json}")
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
    log.info(f"✅ SRT субтитри: {srt_path}")
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
    log.info(f"✅ ASS субтитри: {ass_path}")
    return ass_path

# ─────────────────────────────────────────────
# Step 7 — Burn Subtitles into Video
# ─────────────────────────────────────────────
def burn_subtitles(video_path: Path, srt_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / "video_subtitled.mp4"
    log.info("🎬 Накладання субтитрів через Python (Pillow + MoviePy)...")

    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", "")

    working_path = output_dir / "video_h264.mp4"
    if cfg.CONVERT_TO_1080P:
        if not working_path.exists():
            log.info("🔄 Конвертація HEVC → h264 1080p...")
            subprocess.run(
                f'ffmpeg -y -i "{video_path}" -c:v libx264 -crf 18 -preset fast '
                f'-vf "scale=1080:1920" -c:a aac "{working_path}"',
                shell=True, env=env, capture_output=True
            )
            log.info("✅ Конвертація завершена")
        video = VideoFileClip(str(working_path))
    else:
        log.info("⏩ Конвертація пропущена (CONVERT_TO_1080P=false)")
        video = VideoFileClip(str(video_path))

    subs = pysrt.open(str(srt_path))

    subtitle_clips = []
    for sub in subs:
        start = sub.start.ordinal / 1000.0
        end   = sub.end.ordinal / 1000.0
        duration = end - start

        # Автоматично масштабуємо шрифт відносно висоти відео
        auto_font_size = max(cfg.SUBTITLE_FONT_SIZE, int(video.h * 0.045))

        txt_clip = (
            TextClip(
                text=sub.text.strip(),
                font="/System/Library/Fonts/Supplemental/Arial.ttf",
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

        # Позиція субтитрів
        if cfg.SUBTITLE_POSITION == "top":
            pos = ("center", 50)
        elif cfg.SUBTITLE_POSITION == "center":
            pos = ("center", "center")
        else:  # bottom
            pos = ("center", video.h - txt_clip.h - 60)

        subtitle_clips.append(txt_clip.with_position(pos))

    final = CompositeVideoClip([video, *subtitle_clips])
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    log.info(f"✅ Відео з субтитрами: {output_path}")
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
        log.info("⏩ Форматування пропущено (original)")
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
        f"Форматування відео → {cfg.OUTPUT_FORMAT}",
    )
    log.info(f"✅ Відформатоване відео: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Step 9 — Generate Metadata (Instagram + TikTok)
# ─────────────────────────────────────────────
def generate_metadata(segments: list[dict], output_dir: Path) -> dict:
    if not cfg.ANTHROPIC_API_KEY:
        log.warning("⚠️  ANTHROPIC_API_KEY не задано, пропускаємо генерацію метаданих")
        return {}

    log.info("📋 Генерація метаданих (Instagram + TikTok) через Claude...")
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)

    transcript_preview = " ".join(seg["text"] for seg in segments)[:3000]

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[
            {"role": "user", "content": METADATA_PROMPT},
            {"role": "assistant", "content": "Understood. Please provide the video transcript."},
            {"role": "user", "content": f"Video transcript:\n{transcript_preview}"},
        ],
    )

    raw = message.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("⚠️  Не вдалося розпарсити метадані як JSON")
        metadata = {"raw": raw}

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log.info(f"✅ Метадані збережено: {metadata_path}")
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

AUPHONIC_WATERMARK_KEYWORDS = [
    "auphonic", "biophonic", "phonic", 
    "free audio", "post-production", "post production"
]

def trim_auphonic_watermark(audio_path: Path, output_dir: Path) -> Path:
    """Видаляє Auphonic watermark з початку аудіо."""
    try:
        import whisperx
    except ImportError:
        return audio_path

    log.info("🔍 Пошук Auphonic watermark в аудіо...")

    # Швидка транскрипція тільки перших 15 секунд
    model = whisperx.load_model("base", device=cfg.WHISPER_DEVICE, compute_type="float32")
    audio = whisperx.load_audio(str(audio_path))
    audio_preview = audio[:15 * 16000]  # перші 15 секунд
    result = model.transcribe(audio_preview, language="en", batch_size=16)

    # Шукаємо де закінчується watermark
    cut_time = 0.0
    for seg in result["segments"]:
        text_lower = seg["text"].lower()
        is_watermark = any(kw in text_lower for kw in AUPHONIC_WATERMARK_KEYWORDS)
        if is_watermark:
            cut_time = seg["end"]
            log.info(f"🎯 Знайдено watermark: '{seg['text'].strip()}' → ріжемо до {cut_time:.1f}s")

    if cut_time == 0.0:
        log.info("✅ Watermark не знайдено")
        return audio_path

    # Обрізаємо аудіо
    trimmed_path = output_dir / "audio_trimmed.wav"
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", "")
    subprocess.run(
        f'ffmpeg -y -i "{audio_path}" -ss {cut_time:.3f} -c copy "{trimmed_path}"',
        shell=True, env=env, capture_output=True
    )
    log.info(f"✅ Аудіо обрізано з {cut_time:.1f}s: {trimmed_path}")
    return trimmed_path

# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def run_pipeline(input_video: str, output_dir: str, steps: list[str] = None):
    video_path = Path(input_video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Відео не знайдено: {video_path}")

    out = ensure_dir(Path(output_dir))
    log.info(f"🚀 Починаємо пайплайн для: {video_path.name}")
    log.info(f"📁 Вихідна директорія: {out}")

    all_steps = ["audio", "enhance", "remove_watermark", "merge", "transcribe", "fix", "subtitles", "format", "metadata"]
    active = set(steps) if steps else set(all_steps)
    state  = {"video": video_path}

    # 1. Витяг аудіо
    if "audio" in active:
        state["audio"] = extract_audio(state["video"], out)

    # 2. Покращення звуку
    if "enhance" in active and "audio" in state:
        state["enhanced_audio"] = enhance_audio_auphonic(state["audio"], out)
    else:
        state["enhanced_audio"] = state.get("audio", video_path)

    if "remove_watermark" in active and "audio" in state and cfg.REMOVE_AUPHONIC_WATERMARK:
        state["enhanced_audio"] = trim_auphonic_watermark(state["enhanced_audio"], out)

    # 3. Злиття відео + аудіо
    if "merge" in active and state.get("enhanced_audio") != state.get("audio"):
        state["video"] = merge_audio_video(state["video"], state["enhanced_audio"], out)
    else:
        # Пріоритет: subtitled → enhanced → original
        video_subtitled = out / "video_subtitled.mp4"
        video_enhanced  = out / "video_enhanced.mp4"

        if video_subtitled.exists() and "subtitles" not in active:
            state["video"] = video_subtitled
            log.info(f"📂 Використовуємо існуюче відео з субтитрами: {video_subtitled}")
        elif video_enhanced.exists():
            state["video"] = video_enhanced
            log.info(f"📂 Використовуємо покращене відео: {video_enhanced}")
        else:
            dest = out / video_path.name
            if not dest.exists():
                shutil.copy2(video_path, dest)
            state["video"] = dest
            log.info(f"📂 Використовуємо оригінальне відео: {dest}")
        
    # Завантажуємо існуючий транскрипт якщо є
    fixed_json = out / "transcript_fixed.json"
    raw_json = out / "transcript_raw.json"
    if "transcribe" not in active:
        if fixed_json.exists():
            with open(fixed_json, encoding="utf-8") as f:
                state["segments"] = json.load(f)
            log.info(f"📂 Завантажено існуючий транскрипт: {fixed_json}")
        elif raw_json.exists():
            with open(raw_json, encoding="utf-8") as f:
                state["segments"] = json.load(f)
            log.info(f"📂 Завантажено існуючий транскрипт: {raw_json}")

    # 4. Транскрипція → зберігаємо локально
    if "transcribe" in active:
        audio_src = state.get("enhanced_audio") or state.get("audio")
        state["segments"] = transcribe(audio_src, out)

    # 5. Виправлення тексту
    if "fix" in active and "segments" in state:
        state["segments"] = fix_transcript(state["segments"], out)

    # 6. Субтитри
    if "subtitles" in active and "segments" in state:
        srt_path = generate_srt(state["segments"], out)
        generate_ass(state["segments"], out)
        state["video"] = burn_subtitles(state["video"], srt_path, out)

    # 7. Форматування відео
    if "format" in active:
        state["video"] = format_video(state["video"], out)

    # 8. Метадані Instagram + TikTok
    if "metadata" in active and "segments" in state:
        state["metadata"] = generate_metadata(state["segments"], out)
        print_metadata(state["metadata"])

    log.info("\n" + "=" * 55)
    log.info("🎉 ПАЙПЛАЙН ЗАВЕРШЕНО!")
    log.info(f"📹 Фінальне відео : {state['video']}")
    log.info(f"📁 Всі файли      : {out}")
    return state


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Відео-пайплайн: аудіо → транскрипція → субтитри → метадані"
    )
    parser.add_argument("input", help="Шлях до вхідного відео файлу")
    parser.add_argument("-o", "--output", default="./output",
                        help="Директорія для результатів (default: ./output)")
    parser.add_argument(
        "--steps", nargs="+",
        choices=["audio", "enhance", "remove_watermark", "merge", "transcribe", "fix", "subtitles", "format", "metadata"],
        help="Запустити тільки вказані кроки (default: всі)",
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.steps)


if __name__ == "__main__":
    main()