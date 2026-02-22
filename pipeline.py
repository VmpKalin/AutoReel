#!/usr/bin/env python3
"""
Video Processing Pipeline
=========================
Повний пайплайн для обробки відео:
1. Витяг аудіо
2. Покращення звуку (Auphonic)
3. Транскрипція (WhisperX / Deepgram)
4. Виправлення тексту (Claude API)
5. Генерація субтитрів
6. Накладання субтитрів на відео
7. Форматування відео (рамка, crop)
8. Нарізка Reels
9. Генерація назви, підпису, метаданих
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import subprocess
import requests
from pathlib import Path
from datetime import timedelta
from typing import Optional
from dotenv import load_dotenv
import anthropic

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
    AUPHONIC_USER = os.getenv("AUPHONIC_USER", "")
    AUPHONIC_PASS = os.getenv("AUPHONIC_PASS", "")
    AUPHONIC_PRESET = os.getenv("AUPHONIC_PRESET", "")  # optional preset UUID

    # Deepgram (альтернатива WhisperX якщо немає GPU)
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

    # Whisper
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")  # tiny/base/small/medium/large-v3
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")     # cpu / cuda
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "uk")  # uk / ru / en

    # Subtitles style
    SUBTITLE_FONT = os.getenv("SUBTITLE_FONT", "Arial")
    SUBTITLE_FONT_SIZE = int(os.getenv("SUBTITLE_FONT_SIZE", "18"))
    SUBTITLE_COLOR = os.getenv("SUBTITLE_COLOR", "&H00FFFFFF")  # white
    SUBTITLE_OUTLINE_COLOR = os.getenv("SUBTITLE_OUTLINE_COLOR", "&H00000000")  # black
    SUBTITLE_OUTLINE_SIZE = int(os.getenv("SUBTITLE_OUTLINE_SIZE", "2"))
    SUBTITLE_POSITION = os.getenv("SUBTITLE_POSITION", "bottom")  # bottom / top / center

    # Video format
    OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "16:9")  # 16:9 / 9:16 / 1:1 / original
    ADD_PADDING = os.getenv("ADD_PADDING", "false").lower() == "true"
    PADDING_COLOR = os.getenv("PADDING_COLOR", "black")

    # Reels
    REELS_COUNT = int(os.getenv("REELS_COUNT", "3"))
    REELS_MIN_DURATION = int(os.getenv("REELS_MIN_DURATION", "30"))
    REELS_MAX_DURATION = int(os.getenv("REELS_MAX_DURATION", "60"))

    # Transcription backend: "whisperx" / "deepgram"
    TRANSCRIPTION_BACKEND = os.getenv("TRANSCRIPTION_BACKEND", "whisperx")


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
- Return ONLY the corrected lines in format [N] text
- Nothing else, no explanations"""


REELS_SELECTION_PROMPT = """You are a viral social media video editor specializing in Instagram Reels and TikTok.
Analyze the transcript and find the most engaging moments for short-form content.

What makes a great Reel/TikTok moment:
- Strong hook in the first 3 seconds
- Emotional or surprising moments
- Actionable tips or insights
- Funny or relatable content
- Clear standalone value (no context needed)
- Energetic or passionate delivery

Return ONLY a JSON array, no markdown:
[{"start": 12.5, "end": 45.0, "title": "Short catchy clip title", "hook": "First sentence that grabs attention", "reason": "Why this will perform well"}]"""


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
    """Виконати shell-команду і повернути stdout."""
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
    h = total_ms // 3_600_000
    m = (total_ms % 3_600_000) // 60_000
    s = (total_ms % 60_000) // 1000
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
    if not cfg.AUPHONIC_USER or not cfg.AUPHONIC_PASS:
        log.warning("⚠️  Auphonic credentials не задані, пропускаємо покращення звуку")
        return audio_path

    log.info("🔊 Завантаження аудіо в Auphonic...")
    # auth = (cfg.AUPHONIC_USER, cfg.AUPHONIC_PASS)
    headers = {"Authorization": f"Bearer {cfg.AUPHONIC_API_KEY}"}

    # Збираємо параметри продакшена
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
    production = resp.json()["data"]
    uuid = production["uuid"]
    log.info(f"📤 Production UUID: {uuid}")

    # Polling до завершення
    log.info("⏳ Чекаємо завершення Auphonic (може зайняти хвилину)...")
    for _ in range(120):  # max 10 хвилин
        time.sleep(5)
        status_resp = requests.get(
            f"https://auphonic.com/api/production/{uuid}.json", headers=headers
        )
        status_resp.raise_for_status()
        prod_data = status_resp.json()["data"]
        status_code = prod_data.get("status_string", "")
        log.info(f"   Status: {status_code}")
        if prod_data["status"] == 3:  # Done
            break
        if prod_data["status"] in (9, 10):  # Error / Aborted
            raise RuntimeError(f"Auphonic failed with status: {status_code}")

    # Завантажуємо результат
    output_files = prod_data.get("output_files", [])
    if not output_files:
        raise RuntimeError("Auphonic не повернув файли результату")

    download_url = output_files[0]["download_url"]
    enhanced_path = output_dir / "audio_enhanced.wav"

    log.info(f"📥 Завантаження покращеного аудіо...")
    audio_resp = requests.get(download_url, headers=headers)
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
# Step 4 — Transcription
# ─────────────────────────────────────────────
def transcribe_whisperx(audio_path: Path) -> list[dict]:
    """Транскрипція через WhisperX (локально). Повертає список segments."""
    try:
        import whisperx
        import torch
    except ImportError:
        raise ImportError(
            "WhisperX не встановлено. Виконай: pip install whisperx torch"
        )

    device = cfg.WHISPER_DEVICE
    log.info(f"🎙️  WhisperX транскрипція (model={cfg.WHISPER_MODEL}, device={device})...")

    model = whisperx.load_model(cfg.WHISPER_MODEL, device=device, compute_type="float32")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=cfg.WHISPER_LANGUAGE, batch_size=16)

    # Word-level alignment
    log.info("🔡 Word-level alignment...")
    model_a, metadata = whisperx.load_align_model(
        language_code=cfg.WHISPER_LANGUAGE, device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
    )
    return result["segments"]

def transcribe(audio_path: Path) -> list[dict]:
    return transcribe_whisperx(audio_path)

# ─────────────────────────────────────────────
# Step 5 — Fix Transcript via Claude
# ─────────────────────────────────────────────
def fix_transcript(segments: list[dict]) -> list[dict]:
    """Виправляємо текст через Claude, зберігаємо тайм-коди."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("⚠️  ANTHROPIC_API_KEY not set, skipping transcript fix")
        return segments

    log.info("✍️  Виправлення тексту через Claude...")
    client = anthropic.Anthropic()

    # Збираємо весь текст
    full_text = "\n".join(
        f"[{i}] {seg['text']}" for i, seg in enumerate(segments)
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": TRANSCRIPT_FIX_PROMPT},
            {"role": "assistant", "content": "Understood. Please provide the transcript lines."},
            {"role": "user", "content": full_text},
        ],
    )

    corrected_lines = {}
    for line in message.content[0].text.strip().split("\n"):
        line = line.strip()
        if line.startswith("["):
            try:
                idx_end = line.index("]")
                idx = int(line[1:idx_end])
                text = line[idx_end + 1:].strip()
                corrected_lines[idx] = text
            except (ValueError, IndexError):
                continue

    # Застосовуємо виправлення
    for i, seg in enumerate(segments):
        if i in corrected_lines:
            seg["text"] = corrected_lines[i]

    log.info("✅ Текст виправлено")
    return segments


# ─────────────────────────────────────────────
# Step 6 — Generate SRT
# ─────────────────────────────────────────────
def generate_srt(segments: list[dict], output_dir: Path) -> Path:
    srt_path = output_dir / "subtitles.srt"
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(f"✅ Субтитри збережено: {srt_path}")
    return srt_path


def generate_ass(segments: list[dict], output_dir: Path) -> Path:
    """Генеруємо ASS з підтримкою стилів."""
    ass_path = output_dir / "subtitles.ass"

    # Позиція субтитрів
    alignment_map = {"bottom": 2, "top": 8, "center": 5}
    alignment = alignment_map.get(cfg.SUBTITLE_POSITION, 2)

    header = f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cfg.SUBTITLE_FONT},{cfg.SUBTITLE_FONT_SIZE},{cfg.SUBTITLE_COLOR},&H000000FF,{cfg.SUBTITLE_OUTLINE_COLOR},&H00000000,0,0,0,0,100,100,0,0,1,{cfg.SUBTITLE_OUTLINE_SIZE},0,{alignment},10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for seg in segments:
        start = format_ass_time(seg["start"])
        end = format_ass_time(seg["end"])
        text = seg["text"].strip().replace("\n", "\\N")
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))

    log.info(f"✅ ASS субтитри збережено: {ass_path}")
    return ass_path


# ─────────────────────────────────────────────
# Step 7 — Burn Subtitles into Video
# ─────────────────────────────────────────────
def burn_subtitles(video_path: Path, srt_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / "video_subtitled.mp4"

    # ASS дає кращий контроль за стилем
    ass_path = srt_path.with_suffix(".ass")
    if ass_path.exists():
        subtitle_filter = f"ass='{ass_path}'"
    else:
        style = (
            f"FontName={cfg.SUBTITLE_FONT},"
            f"FontSize={cfg.SUBTITLE_FONT_SIZE},"
            f"PrimaryColour={cfg.SUBTITLE_COLOR},"
            f"OutlineColour={cfg.SUBTITLE_OUTLINE_COLOR},"
            f"Outline={cfg.SUBTITLE_OUTLINE_SIZE}"
        )
        subtitle_filter = f"subtitles='{srt_path}':force_style='{style}'"

    run(
        f'ffmpeg -y -i "{video_path}" -vf "{subtitle_filter}" '
        f'-c:a copy "{output_path}"',
        "Накладання субтитрів на відео",
    )
    log.info(f"✅ Відео з субтитрами: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Step 8 — Format Video (padding / crop)
# ─────────────────────────────────────────────
ASPECT_RATIOS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}


def format_video(video_path: Path, output_dir: Path, suffix: str = "formatted") -> Path:
    if cfg.OUTPUT_FORMAT == "original" and not cfg.ADD_PADDING:
        log.info("⏩ Форматування пропущено (original)")
        return video_path

    output_path = output_dir / f"video_{suffix}.mp4"
    target = ASPECT_RATIOS.get(cfg.OUTPUT_FORMAT, (1920, 1080))
    w, h = target

    if cfg.ADD_PADDING:
        # Scale + pad (додаємо поля)
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:{cfg.PADDING_COLOR}"
        )
    else:
        # Scale + crop (обрізаємо)
        vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"

    run(
        f'ffmpeg -y -i "{video_path}" -vf "{vf}" -c:a copy "{output_path}"',
        f"Форматування відео → {cfg.OUTPUT_FORMAT}",
    )
    log.info(f"✅ Відформатоване відео: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Step 9 — Generate Reels via Claude
# ─────────────────────────────────────────────
def get_reels_timestamps(segments: list[dict]) -> list[dict]:
    """Просимо Claude вибрати найкращі моменти для Reels."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("⚠️  ANTHROPIC_API_KEY not set, using fallback reel splitting")
        return _fallback_reels(segments)

    log.info("🎬 Генерація тайм-кодів для Reels через Claude...")
    client = anthropic.Anthropic()

    transcript_with_times = "\n".join(
        f"[{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}"
        for seg in segments
    )

    user_prompt = (
        f"Here is the video transcript with timestamps:\n{transcript_with_times}\n\n"
        f"Find the {cfg.REELS_COUNT} best moments, each between {cfg.REELS_MIN_DURATION} and "
        f"{cfg.REELS_MAX_DURATION} seconds long."
    )
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": REELS_SELECTION_PROMPT},
            {"role": "assistant", "content": "Understood. Please provide the transcript with timestamps."},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = message.content[0].text.strip()
    # Видаляємо можливі markdown-теги
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        reels = json.loads(raw)
        log.info(f"✅ Claude запропонував {len(reels)} Reels")
        return reels
    except json.JSONDecodeError:
        log.warning("⚠️  Не вдалося розпарсити JSON від Claude, використовуємо fallback")
        return _fallback_reels(segments)


def _fallback_reels(segments: list[dict]) -> list[dict]:
    """Рівномірна нарізка якщо Claude недоступний."""
    if not segments:
        return []
    total_duration = segments[-1]["end"]
    step = total_duration / cfg.REELS_COUNT
    reels = []
    for i in range(cfg.REELS_COUNT):
        start = i * step
        end = min(start + cfg.REELS_MAX_DURATION, total_duration)
        reels.append({"start": start, "end": end, "title": f"Reel {i+1}", "reason": "auto"})
    return reels


def cut_reels(video_path: Path, reels: list[dict], output_dir: Path) -> list[Path]:
    reels_dir = ensure_dir(output_dir / "reels")
    reel_paths = []
    for i, reel in enumerate(reels, 1):
        start = reel["start"]
        end = reel["end"]
        title_slug = reel.get("title", f"reel_{i}").replace(" ", "_")[:30]
        out = reels_dir / f"reel_{i:02d}_{title_slug}.mp4"

        run(
            f'ffmpeg -y -i "{video_path}" -ss {start:.2f} -to {end:.2f} -c copy "{out}"',
            f"Нарізка Reel {i}: {reel.get('title', '')}",
        )
        reel_paths.append(out)

    # Форматуємо Reels у 9:16 якщо потрібно
    if cfg.OUTPUT_FORMAT != "original":
        formatted_reels = []
        for reel_path in reel_paths:
            formatted = format_video(reel_path, reels_dir, suffix=reel_path.stem + "_9x16")
            formatted_reels.append(formatted)
        return formatted_reels

    log.info(f"✅ Нарізано {len(reel_paths)} Reels")
    return reel_paths


# ─────────────────────────────────────────────
# Step 10 — Generate Metadata via Claude
# ─────────────────────────────────────────────
def generate_metadata(segments: list[dict], output_dir: Path) -> dict:
    if not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("⚠️  ANTHROPIC_API_KEY not set, skipping metadata generation")
        return {}

    log.info("📋 Генерація метаданих через Claude...")
    client = anthropic.Anthropic()

    # Беремо перші 3000 символів транскрипту
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

    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("⚠️  Не вдалося розпарсити метадані")
        metadata = {"raw": raw}

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log.info(f"✅ Метадані збережено: {metadata_path}")
    return metadata


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

    all_steps = ["audio", "enhance", "merge", "transcribe", "fix", "subtitles", "format", "reels", "metadata"]
    active = set(steps) if steps else set(all_steps)

    # Зберігаємо стан між кроками
    state = {"video": video_path}

    # 1. Витяг аудіо
    if "audio" in active:
        state["audio"] = extract_audio(state["video"], out)

    # 2. Покращення звуку
    if "enhance" in active and "audio" in state:
        state["enhanced_audio"] = enhance_audio_auphonic(state["audio"], out)
    else:
        state["enhanced_audio"] = state.get("audio", video_path)

    # 3. Злиття відео + аудіо
    if "merge" in active and state["enhanced_audio"] != state.get("audio"):
        state["video"] = merge_audio_video(state["video"], state["enhanced_audio"], out)
    else:
        # Копіюємо вхідне відео у out для консистентності
        dest = out / video_path.name
        if not dest.exists():
            shutil.copy2(video_path, dest)
        state["video"] = dest

    # 4. Транскрипція
    if "transcribe" in active:
        state["segments"] = transcribe(state.get("enhanced_audio", state["audio"]))
        # Зберігаємо сирий транскрипт
        raw_path = out / "transcript_raw.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(state["segments"], f, ensure_ascii=False, indent=2)
        log.info(f"💾 Сирий транскрипт: {raw_path}")

    # 5. Виправлення тексту
    if "fix" in active and "segments" in state:
        state["segments"] = fix_transcript(state["segments"])
        fixed_path = out / "transcript_fixed.json"
        with open(fixed_path, "w", encoding="utf-8") as f:
            json.dump(state["segments"], f, ensure_ascii=False, indent=2)

    # 6. Субтитри
    if "subtitles" in active and "segments" in state:
        srt_path = generate_srt(state["segments"], out)
        ass_path = generate_ass(state["segments"], out)
        state["video"] = burn_subtitles(state["video"], srt_path, out)

    # 7. Форматування відео
    if "format" in active:
        state["video"] = format_video(state["video"], out)

    # 8. Reels
    if "reels" in active and "segments" in state:
        reels_timestamps = get_reels_timestamps(state["segments"])
        reels_info_path = out / "reels_timestamps.json"
        with open(reels_info_path, "w", encoding="utf-8") as f:
            json.dump(reels_timestamps, f, ensure_ascii=False, indent=2)
        state["reels"] = cut_reels(state["video"], reels_timestamps, out)

    # 9. Метадані
    if "metadata" in active and "segments" in state:
        state["metadata"] = generate_metadata(state["segments"], out)
        if state["metadata"]:
            print("\n" + "=" * 50)
            print("📋 GENERATED METADATA:")
            print("=" * 50)
            list_fields = {"instagram_hashtags", "tiktok_hashtags"}
            for k, v in state["metadata"].items():
                if k in list_fields:
                    print(f"\n🔹 {k.upper()}: {', '.join(v)}")
                else:
                    print(f"\n🔹 {k.upper()}:\n{v}")

    # Фінальний звіт
    final_video = state["video"]
    log.info("\n" + "=" * 50)
    log.info("🎉 ПАЙПЛАЙН ЗАВЕРШЕНО!")
    log.info(f"📹 Фінальне відео: {final_video}")
    log.info(f"📁 Всі файли: {out}")
    if "reels" in state:
        log.info(f"✂️  Reels: {len(state['reels'])} файлів у {out / 'reels'}")

    return state


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Повний відео-пайплайн: аудіо → транскрипція → субтитри → Reels"
    )
    parser.add_argument("input", help="Шлях до вхідного відео файлу")
    parser.add_argument(
        "-o", "--output", default="./output", help="Директорія для результатів (default: ./output)"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["audio", "enhance", "merge", "transcribe", "fix", "subtitles", "format", "reels", "metadata"],
        help="Запустити тільки вказані кроки (default: всі)",
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.steps)


if __name__ == "__main__":
    main()
