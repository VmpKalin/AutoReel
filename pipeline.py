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
import unicodedata
import argparse
import subprocess
import requests
import ollama
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv

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

    # Transcription backend: whisperx (transcribe+align) | hybrid (OpenAI text + WhisperX align)
    TRANSCRIPTION_BACKEND   = os.getenv("TRANSCRIPTION_BACKEND", "whisperx").lower()
    OPENAI_API_KEY          = os.getenv("OPENAI_API_KEY", "")
    OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
    # Hybrid alignment sanity guard (drift detection on force-aligned word timings)
    WORD_MAX_DURATION       = float(os.getenv("WORD_MAX_DURATION", "2.0"))  # implausible single-word length (s)
    ALIGN_CLAMP_LONG_WORDS  = os.getenv("ALIGN_CLAMP_LONG_WORDS", "true").lower() == "true"

    # Subtitles
    SUBTITLE_FONT_PATH     = os.getenv("SUBTITLE_FONT_PATH", "/System/Library/Fonts/Supplemental/Impact.ttf")
    SUBTITLE_FONT          = os.getenv("SUBTITLE_FONT", "Impact")
    SUBTITLE_FONT_SIZE     = int(os.getenv("SUBTITLE_FONT_SIZE", "18"))
    SUBTITLE_COLOR         = os.getenv("SUBTITLE_COLOR", "&H0000FFFF")          # yellow (ASS &HAABBGGRR, opaque)
    SUBTITLE_OUTLINE_COLOR = os.getenv("SUBTITLE_OUTLINE_COLOR", "&H00000000")  # black, thin edge
    SUBTITLE_OUTLINE_SIZE  = int(os.getenv("SUBTITLE_OUTLINE_SIZE", "1"))       # thin outline (legibility only)
    # Shadow lifts the text off the video instead of a chunky border.
    # ASS alpha is INVERTED: 00 = fully opaque, FF = fully invisible, 80 ≈ 50% transparent.
    # Do not "fix" the 80 in the default to 00 — that would make the shadow solid black.
    SUBTITLE_SHADOW_COLOR  = os.getenv("SUBTITLE_SHADOW_COLOR", "&H80000000")   # ~50% transparent black
    SUBTITLE_SHADOW_SIZE   = int(os.getenv("SUBTITLE_SHADOW_SIZE", "2"))        # drop-shadow depth
    SUBTITLE_POSITION      = os.getenv("SUBTITLE_POSITION", "bottom")
    SUBTITLE_BOTTOM_MARGIN = int(os.getenv("SUBTITLE_BOTTOM_MARGIN", "80"))
    SUBTITLE_WORDS_PER_CAPTION = int(os.getenv("SUBTITLE_WORDS_PER_CAPTION", "2"))
    SUBTITLE_MAX_CHARS     = int(os.getenv("SUBTITLE_MAX_CHARS", "16"))  # width proxy: split long captions to fewer words
    SUBTITLE_UPPERCASE     = os.getenv("SUBTITLE_UPPERCASE", "true").lower() == "true"
    SUBTITLE_BOLD          = os.getenv("SUBTITLE_BOLD", "true").lower() == "true"
    SUBTITLE_STRIP_PUNCTUATION = os.getenv("SUBTITLE_STRIP_PUNCTUATION", "true").lower() == "true"

    # Video format
    OUTPUT_FORMAT    = os.getenv("OUTPUT_FORMAT", "9:16")
    ADD_PADDING      = os.getenv("ADD_PADDING", "false").lower() == "true"
    PADDING_COLOR    = os.getenv("PADDING_COLOR", "black")
    CONVERT_TO_1080P = os.getenv("CONVERT_TO_1080P", "true").lower() == "true"

    VIDEO_SPEED = float(os.getenv("VIDEO_SPEED", "1.0"))  # 1.0 = normal, 1.5 = 1.5x, 2.0 = 2x

    # Filler-word removal (слова-паразити)
    FILLER_WORDS          = os.getenv("FILLER_WORDS", "")              # extra fillers, comma-separated (extends defaults + filler_words.txt)
    FILLER_AGGRESSIVE     = os.getenv("FILLER_AGGRESSIVE", "false").lower() == "true"  # also cut risky homographs (ну, от, та, там)
    FILLER_AUDIO_XFADE_MS = int(os.getenv("FILLER_AUDIO_XFADE_MS", "40"))   # audio declick fade length at each join
    FILLER_VIDEO_XFADE    = os.getenv("FILLER_VIDEO_XFADE", "false").lower() == "true"  # v1: honored as a warning only (hard-cut)
    FILLER_VIDEO_XFADE_MS = int(os.getenv("FILLER_VIDEO_XFADE_MS", "120"))

cfg = Config()


# ─────────────────────────────────────────────
# Default filler lists (Ukrainian "слова-паразити")
# ─────────────────────────────────────────────
# SAFE: distinctive, almost always filler — cut by default. Multi-word entries
# match consecutive words. Normalized (lowercased, punctuation-stripped) before compare.
FILLER_DEFAULT_SAFE = [
    "еее", "ееє", "еем", "ееем", "ммм", "емм",
    "типу", "тіпа", "коротше", "значить", "блін",
    "як би", "якби", "в общем", "ну та", "оце от", "власне",
]
# AGGRESSIVE: short homographs that are often LEGITIMATE words (от = "вот/ось",
# та = "and/but", там = "there", ну = discourse marker). Only cut when
# FILLER_AGGRESSIVE=true, since these cause false positives.
FILLER_DEFAULT_AGGRESSIVE = ["ну", "от", "та", "там"]

# Cut intervals closer than this are merged into one join (avoids micro-segments).
FILLER_MERGE_GAP = 0.08  # seconds


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


def resolve_output_dir(requested_output_dir: str | Path, allow_existing: bool = False) -> Path:
    base_dir = Path(requested_output_dir).resolve()
    ensure_dir(base_dir)

    # Resume runs (partial --steps) reuse the dir so hand-edited files like
    # words_edit.json are found instead of forking off to a fresh _N copy.
    if allow_existing or not any(item.is_file() for item in base_dir.iterdir()):
        return base_dir

    parent = base_dir.parent
    stem = base_dir.name
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}"
        if not candidate.exists():
            ensure_dir(candidate)
            log.info(f"📁 Output directory has files, using: {candidate}")
            return candidate
        if candidate.is_dir() and not any(item.is_file() for item in candidate.iterdir()):
            log.info(f"📁 Output directory has files, using: {candidate}")
            return candidate
        idx += 1


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
def _wx_align(segments: list[dict], audio, device) -> list[dict]:
    """Force-align given text segments onto audio → word-level start/end via WhisperX."""
    import whisperx
    model_a, metadata = whisperx.load_align_model(
        language_code=cfg.WHISPER_LANGUAGE, device=device
    )
    result = whisperx.align(
        segments, model_a, metadata, audio, device, return_char_alignments=False,
    )
    return result.get("word_segments", [])


def _transcribe_whisperx(audio_path: Path) -> list[dict]:
    """Pure WhisperX path: WhisperX transcribes the audio AND aligns it (unchanged)."""
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
    word_segments = _wx_align(result["segments"], audio, device)
    if not word_segments:
        # No word-level alignment — fall back to segment-level "words".
        word_segments = [
            {"word": s.get("text", "").strip(), "start": s.get("start"), "end": s.get("end")}
            for s in result.get("segments", [])
            if s.get("text", "").strip()
        ]
    return word_segments


def _openai_transcribe_text(audio_path: Path) -> str:
    """Get high-accuracy TEXT from OpenAI gpt-4o-transcribe (no word timestamps).

    Endpoint: POST /v1/audio/transcriptions (SDK: client.audio.transcriptions.create).
    Params: model=OPENAI_TRANSCRIBE_MODEL, language=WHISPER_LANGUAGE (ISO 639-1),
    response_format="json" (gpt-4o-transcribe supports only json|text, NOT verbose_json).
    """
    if not cfg.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set — required for TRANSCRIPTION_BACKEND=hybrid.")

    # OpenAI transcription endpoint caps uploads at 25 MB. No chunking in v1.
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb > 25:
        raise RuntimeError(
            f"Audio is {size_mb:.1f} MB > OpenAI's 25 MB limit. v1 has no chunking — "
            f"use TRANSCRIPTION_BACKEND=whisperx, or shorten/compress the clip."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai SDK not installed. Run: pip install openai")

    client = OpenAI(api_key=cfg.OPENAI_API_KEY)
    log.info(f"🤖 OpenAI transcription (model={cfg.OPENAI_TRANSCRIBE_MODEL}, lang={cfg.WHISPER_LANGUAGE})...")
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=cfg.OPENAI_TRANSCRIBE_MODEL,
            file=f,
            language=cfg.WHISPER_LANGUAGE,   # ISO 639-1 hint, e.g. "uk"
            response_format="json",          # json|text only for gpt-4o-transcribe
        )
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty transcript.")
    log.info(f"📝 GPT transcript: {len(text)} chars")
    return text


def _split_into_sentences(text: str, max_words: int = 12) -> list[str]:
    """Split GPT text into sentence-sized chunks to anchor forced alignment.

    Splits on . ! ? … boundaries; over-long sentences are further split on commas,
    then hard-chunked to max_words. Anchors throughout the clip prevent the
    single-span Viterbi drift that smeared words across silence.
    """
    pieces = [p for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()]
    chunks: list[str] = []
    for p in pieces:
        if len(p.split()) <= max_words:
            chunks.append(p.strip())
            continue
        # Long sentence → split on commas first.
        for sub in re.split(r",\s*", p):
            words = sub.split()
            if not words:
                continue
            if len(words) <= max_words:
                chunks.append(sub.strip())
            else:
                for i in range(0, len(words), max_words):
                    chunks.append(" ".join(words[i:i + max_words]))
    return [c for c in chunks if c.strip()] or ([text.strip()] if text.strip() else [])


def _guard_word_durations(word_segments: list[dict]) -> list[dict]:
    """Flag (and optionally clamp) implausibly long word durations from drift.

    Surfaces drift in the logs rather than silently emitting 22s captions. Clamping
    (ALIGN_CLAMP_LONG_WORDS) caps a bad word at start+WORD_MAX_DURATION and logs each.
    """
    thr = cfg.WORD_MAX_DURATION
    long_words = []
    for w in word_segments:
        s, e = w.get("start"), w.get("end")
        if s is None or e is None:
            continue
        dur = e - s
        if dur > thr:
            long_words.append((w.get("word", ""), s, e, dur))
            if cfg.ALIGN_CLAMP_LONG_WORDS:
                w["end"] = round(s + thr, 3)

    if long_words:
        action = "clamped" if cfg.ALIGN_CLAMP_LONG_WORDS else "left as-is"
        log.warning(
            f"⚠️  {len(long_words)} word(s) exceed WORD_MAX_DURATION={thr}s "
            f"(possible alignment drift; {action}):"
        )
        for word, s, e, dur in long_words:
            log.warning(f"     {format_srt_time(s)}–{format_srt_time(e)} ({dur:.1f}s) \"{word}\"")
        log.warning("     If drift is widespread, the GPT text likely diverges from the "
                    "audio — consider TRANSCRIPTION_BACKEND=whisperx for this content.")
    return word_segments


def _transcribe_hybrid(audio_path: Path) -> list[dict]:
    """Hybrid path: GPT-4o-transcribe TEXT + WhisperX forced alignment for word timings.

    WhisperX does NOT transcribe here — it only aligns the GPT text onto the audio,
    yielding the same {word, start, end} structure the rest of the pipeline expects.
    """
    text = _openai_transcribe_text(audio_path)

    try:
        import whisperx
    except ImportError:
        raise ImportError("WhisperX is not installed. Run: pip install whisperx torch")

    device = cfg.WHISPER_DEVICE
    audio  = whisperx.load_audio(str(audio_path))
    duration = len(audio) / 16000.0  # whisperx loads at 16 kHz

    # Sentence-segment the transcript and give each chunk a rough proportional time
    # window (by word count). These anchors keep alignment on track across the whole
    # clip instead of one giant span that drifts at pauses/paraphrases.
    sentences = _split_into_sentences(text)
    word_counts = [max(1, len(s.split())) for s in sentences]
    total_words = sum(word_counts)
    segments = []
    cursor = 0
    for sent, wc in zip(sentences, word_counts):
        start = duration * (cursor / total_words)
        end   = duration * ((cursor + wc) / total_words)
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": sent})
        cursor += wc

    log.info(f"🔡 Forced alignment of GPT text with WhisperX ({len(segments)} segment anchors)...")
    word_segments = _wx_align(segments, audio, device)

    if not word_segments:
        log.warning("Alignment produced no word timings — emitting one untimed segment.")
        return [{"word": text}]

    return _guard_word_durations(word_segments)


def transcribe(audio_path: Path, output_dir: Path) -> list[dict]:
    backend = cfg.TRANSCRIPTION_BACKEND
    if backend == "hybrid":
        word_segments = _transcribe_hybrid(audio_path)
    elif backend == "whisperx":
        word_segments = _transcribe_whisperx(audio_path)
    else:
        log.warning(f"Unknown TRANSCRIPTION_BACKEND={backend!r}; falling back to whisperx.")
        word_segments = _transcribe_whisperx(audio_path)

    # Persist RAW word-level data. No grouping here — grouping moves to subtitle time
    # so words can be hand-edited (typos) in between without touching timestamps.
    words = []
    for w in word_segments:
        entry = {"word": w.get("word", "")}
        # Keep real timestamps when present; words without them are kept too.
        if w.get("start") is not None:
            entry["start"] = w["start"]
        if w.get("end") is not None:
            entry["end"] = w["end"]
        words.append(entry)

    log.info(f"✅ Transcription: {len(words)} words")

    raw_json = output_dir / "words_raw.json"
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    # Editable copy — fix the `word` fields by hand here; start/end stay for grouping.
    # Transcribe is the source of truth, so this is (re)written on every transcribe run.
    edit_json = output_dir / "words_edit.json"
    with open(edit_json, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    # Read-only plain-text preview of the words.
    raw_txt = output_dir / "transcript.txt"
    with open(raw_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(w["word"] for w in words).strip())

    log.info(f"💾 Raw words: {raw_json}")
    log.info(f"✏️  Edit words here, then run --steps subtitles: {edit_json}")
    return words


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
# Step 5b — Remove filler words (слова-паразити)
# ─────────────────────────────────────────────
def get_media_duration(path: Path) -> float:
    """Total duration (seconds) of a media file via ffprobe."""
    out = run(
        f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{path}"',
        "Probe media duration",
    )
    first = next((l for l in out.splitlines() if l.strip()), "")
    return float(first.strip())


def _norm_word(w: str) -> str:
    """Normalize a word for filler matching: punctuation-stripped + lowercased."""
    return _strip_punctuation(str(w)).lower().strip()


def load_filler_phrases(output_dir: Path) -> list[list[str]]:
    """Build the active filler list as normalized token lists (supports multi-word).

    Sources, all merged: SAFE defaults, AGGRESSIVE defaults (only if enabled),
    an optional filler_words.txt (one phrase per line), and FILLER_WORDS env.
    Longer phrases are checked first so "ну та" wins over "ну".
    """
    raw = list(FILLER_DEFAULT_SAFE)
    if cfg.FILLER_AGGRESSIVE:
        raw += FILLER_DEFAULT_AGGRESSIVE

    for candidate in (Path("filler_words.txt"), Path(__file__).parent / "filler_words.txt"):
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as f:
                raw += [ln for ln in f.read().splitlines() if ln.strip()]
            log.info(f"📄 Extended fillers from {candidate}")
            break

    raw += [p for p in cfg.FILLER_WORDS.split(",") if p.strip()]

    phrases = []
    seen = set()
    for p in raw:
        tokens = [_norm_word(t) for t in p.split() if _norm_word(t)]
        key = " ".join(tokens)
        if tokens and key not in seen:
            seen.add(key)
            phrases.append(tokens)
    # Longest first → greedy multi-word match precedence.
    phrases.sort(key=len, reverse=True)
    return phrases


def _find_filler_cuts(words: list[dict], phrases: list[list[str]]):
    """Scan words, return (cuts, kept_words, report).

    cuts: list of (start, end) intervals to remove (only timestamped matches).
    kept_words: words NOT matched as filler (still in original timeline).
    report: list of {"word", "start", "end"} for what was cut.
    """
    norms = [_norm_word(w.get("word", "")) for w in words]
    n = len(words)
    matched = [False] * n

    i = 0
    while i < n:
        hit = None
        for tokens in phrases:  # already longest-first
            L = len(tokens)
            if i + L <= n and norms[i:i + L] == tokens:
                hit = L
                break
        if hit:
            span = words[i:i + hit]
            starts = [w["start"] for w in span if w.get("start") is not None]
            ends   = [w["end"]   for w in span if w.get("end")   is not None]
            # Only cut when we have real timestamps; otherwise leave the words be.
            if starts and ends:
                for j in range(i, i + hit):
                    matched[j] = True
                i += hit
                continue
        i += 1

    cuts = []
    report = []
    # Re-walk to collect contiguous matched spans as cut intervals + report rows.
    i = 0
    while i < n:
        if matched[i]:
            j = i
            while j < n and matched[j]:
                j += 1
            span = words[i:j]
            s = min(w["start"] for w in span if w.get("start") is not None)
            e = max(w["end"]   for w in span if w.get("end")   is not None)
            cuts.append((s, e))
            report.append({"word": " ".join(w.get("word", "") for w in span), "start": s, "end": e})
            i = j
        else:
            i += 1

    kept_words = [w for k, w in enumerate(words) if not matched[k]]
    return cuts, kept_words, report


def _merge_intervals(cuts, gap=FILLER_MERGE_GAP):
    """Merge overlapping/adjacent cut intervals (separated by < gap)."""
    if not cuts:
        return []
    cuts = sorted(cuts)
    merged = [list(cuts[0])]
    for s, e in cuts[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _keep_segments(cuts, total):
    """Complement of cut intervals over [0, total] → list of (start, end) to keep."""
    keeps = []
    cursor = 0.0
    for s, e in cuts:
        s = max(0.0, min(s, total))
        e = max(0.0, min(e, total))
        if s - cursor > 1e-3:
            keeps.append((cursor, s))
        cursor = max(cursor, e)
    if total - cursor > 1e-3:
        keeps.append((cursor, total))
    return keeps


def _retime_words(kept_words, cuts):
    """Shift each kept word earlier by the total cut duration occurring before it.

    Kept words never lie inside a cut, so start/end share the same offset.
    """
    cuts = sorted(cuts)
    out = []
    for w in kept_words:
        nw = dict(w)
        st = w.get("start")
        if st is not None:
            before = sum((min(e, st) - s) for s, e in cuts if s < st)
            nw["start"] = round(st - before, 3)
            if w.get("end") is not None:
                nw["end"] = round(w["end"] - before, 3)
        out.append(nw)
    return out


def remove_fillers(words: list[dict], video_path: Path, output_dir: Path) -> tuple[Path, list[dict]]:
    """Cut filler-word time ranges from the video and re-time the word list.

    v1 strategy (chosen): HARD-CUT video for frame-accurate timing, with a short
    AUDIO declick fade at each join. Audio length stays equal to video length
    (fades do not overlap/shorten), so video/audio/words never drift. Re-timing
    is exact: subtract the cut duration that precedes each kept word.

    Returns (cleaned_video_path, cleaned_words). If nothing matches, returns the
    original video and words unchanged.
    """
    phrases = load_filler_phrases(output_dir)
    cuts, kept_words, report = _find_filler_cuts(words, phrases)
    cuts = _merge_intervals(cuts)

    # Always write a report (even if empty) so the run is auditable.
    report_json = output_dir / "fillers_report.json"
    report_txt  = output_dir / "fillers_report.txt"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(report_txt, "w", encoding="utf-8") as f:
        for r in report:
            f.write(f"[{format_srt_time(r['start'])} → {format_srt_time(r['end'])}] \"{r['word']}\"\n")
    log.info(f"🗒️  Filler report: {report_txt} ({len(report)} cuts)")

    if cfg.FILLER_VIDEO_XFADE:
        log.warning("FILLER_VIDEO_XFADE=true ignored in v1 — using frame-accurate hard cut "
                    "(video xfade would shorten duration and desync timing).")

    if not cuts:
        log.info("✅ No filler words found — video unchanged.")
        cleaned_words = list(words)
        json.dump(cleaned_words, open(output_dir / "words_cleaned.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        return video_path, cleaned_words

    total = get_media_duration(video_path)
    keeps = _keep_segments(cuts, total)
    if not keeps:
        log.warning("Cuts cover the whole video — skipping filler removal.")
        return video_path, list(words)

    cut_total = sum(e - s for s, e in cuts)
    log.info(f"✂️  Removing {len(cuts)} filler range(s), {cut_total:.2f}s total, "
             f"from {len(keeps)} kept segment(s).")

    # Build a filter_complex: trim each keep segment on BOTH streams (same range →
    # inherently in sync), declick audio with short fades, then concat.
    fade = max(0.0, cfg.FILLER_AUDIO_XFADE_MS / 1000.0)
    parts = []
    labels = []
    for idx, (s, e) in enumerate(keeps):
        dur = e - s
        d = min(fade, dur / 2)  # clamp so fades fit short segments
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{idx}]")
        afilters = [f"atrim=start={s:.3f}:end={e:.3f}", "asetpts=PTS-STARTPTS"]
        if idx != 0 and d > 0:                       # fade in at internal joins
            afilters.append(f"afade=t=in:st=0:d={d:.3f}")
        if idx != len(keeps) - 1 and d > 0:          # fade out before internal joins
            afilters.append(f"afade=t=out:st={max(0.0, dur - d):.3f}:d={d:.3f}")
        parts.append(f"[0:a]{','.join(afilters)}[a{idx}]")
        labels.append(f"[v{idx}][a{idx}]")
    parts.append(f"{''.join(labels)}concat=n={len(keeps)}:v=1:a=1[v][a]")
    filtergraph = ";\n".join(parts)

    filter_file = output_dir / "filler_filter.txt"
    with open(filter_file, "w", encoding="utf-8") as f:
        f.write(filtergraph)

    cleaned_video = output_dir / "video_cleaned.mp4"
    run(
        f'ffmpeg -y -i "{video_path}" -filter_complex_script "{filter_file}" '
        f'-map "[v]" -map "[a]" -c:v libx264 -preset medium -crf 18 '
        f'-c:a aac -b:a 192k "{cleaned_video}"',
        "Cut filler ranges (hard-cut video + audio declick)",
    )

    cleaned_words = _retime_words(kept_words, cuts)
    json.dump(cleaned_words, open(output_dir / "words_cleaned.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # Sanity: last kept word end should be ≈ new (shorter) video duration.
    new_dur = get_media_duration(cleaned_video)
    if cleaned_words:
        last_end = max((w.get("end", 0) for w in cleaned_words), default=0)
        drift = abs(new_dur - last_end)
        log.info(f"⏱️  New duration {new_dur:.2f}s, last word end {last_end:.2f}s (drift {drift:.2f}s).")
    log.info(f"💾 Cleaned video: {cleaned_video}")
    return cleaned_video, cleaned_words


# ─────────────────────────────────────────────
# Step 6 — Generate Subtitles (SRT + ASS)
# ─────────────────────────────────────────────
# Apostrophes are part of Ukrainian words (e.g. "об'єкт"), so they are the one
# punctuation class we KEEP. Straight ', typographic ' and ʼ all count.
_KEEP_PUNCT = {"'", "’", "ʼ"}


def _strip_punctuation(text: str) -> str:
    """Remove all Unicode punctuation except the Ukrainian apostrophe.

    Uses unicodedata category (anything starting with "P") so it also catches
    smart/Cyrillic punctuation („ " — … » etc.), not just ASCII. Doubled spaces
    left behind are collapsed.
    """
    kept = [
        ch for ch in text
        if ch in _KEEP_PUNCT or not unicodedata.category(ch).startswith("P")
    ]
    return re.sub(r"\s{2,}", " ", "".join(kept)).strip()


def _caption_text(text: str) -> str:
    """On-screen caption text, shared by SRT and ASS so both stay consistent.

    Finalization (punctuation strip + uppercase) lives here, not at transcribe
    time, so it survives the Ollama fix step and applies identically to both
    formats. `.upper()` handles Ukrainian Cyrillic correctly.
    """
    text = text.strip()
    if cfg.SUBTITLE_STRIP_PUNCTUATION:
        text = _strip_punctuation(text)
    if cfg.SUBTITLE_UPPERCASE:
        text = text.upper()
    return text


def group_words_into_captions(words: list[dict]) -> list[dict]:
    """Group raw word entries into single-line captions for burning.

    Rules, in priority order:
      1. Target SUBTITLE_WORDS_PER_CAPTION words per caption.
      2. Close early if adding the next word would exceed SUBTITLE_MAX_CHARS.
         (Char count is a proxy for rendered pixel width — good enough; exact
         would need font metrics.) So long words yield fewer-word captions.
      3. A single word longer than the cap is emitted alone — never dropped,
         never hard-split.
    Each caption keeps real word-level timestamps: start = first word's start,
    end = last word's end. Width is measured on the UPPERCASED form (caps are
    wider) so it reflects what's actually rendered.
    """
    segments: list[dict] = []
    if not words:
        return segments

    words_per = max(1, cfg.SUBTITLE_WORDS_PER_CAPTION)
    max_chars = max(1, cfg.SUBTITLE_MAX_CHARS)

    chunk_words: list[str] = []
    chunk_text  = ""
    chunk_start = None
    chunk_end   = None

    def flush():
        nonlocal chunk_words, chunk_text, chunk_start, chunk_end
        if not chunk_words:
            return
        s = chunk_start if chunk_start is not None else words[0].get("start", 0.0)
        e = chunk_end   if chunk_end   is not None else words[-1].get("end", s + 1)
        segments.append({"start": s, "end": e, "text": " ".join(chunk_words)})
        chunk_words = []; chunk_text = ""; chunk_start = None; chunk_end = None

    for word in words:
        w = word.get("word", "")
        has_ts = (word.get("start") is not None) and (word.get("end") is not None)
        w_upper = w.upper()

        candidate = (chunk_text + " " + w_upper).strip() if chunk_text else w_upper

        # Char cap: overflow closes the current caption early (down to 1 word).
        if chunk_words and len(candidate) > max_chars:
            flush()
            candidate = w_upper

        chunk_words.append(w)
        chunk_text = candidate
        if has_ts:
            if chunk_start is None:
                chunk_start = word["start"]
            chunk_end = word["end"]

        # Word-count cap, whichever happens first.
        if len(chunk_words) >= words_per and chunk_start is not None and chunk_end is not None:
            flush()

    flush()
    return segments


def generate_srt(segments: list[dict], output_dir: Path) -> Path:
    srt_path = output_dir / "subtitles.srt"
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(
            f"{i}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n{_caption_text(seg['text'])}\n"
        )
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"✅ SRT subtitles: {srt_path}")
    return srt_path


def generate_ass(segments: list[dict], output_dir: Path, play_res_x: int, play_res_y: int) -> Path:
    ass_path = output_dir / "subtitles.ass"
    alignment = {"bottom": 2, "top": 8, "center": 5}.get(cfg.SUBTITLE_POSITION, 2)

    # Colours come straight from config in ASS &HAABBGGRR (BGR) format — no conversion needed.
    # SecondaryColour (karaoke fill, unused here) mirrors the primary colour so it can't
    # surface as an unexpected tint. OutlineColour is driven by cfg.SUBTITLE_OUTLINE_COLOR
    # (thin edge), and BackColour by cfg.SUBTITLE_SHADOW_COLOR (semi-transparent drop shadow).
    # MarginV uses SUBTITLE_BOTTOM_MARGIN to keep text inside the lower safe area.
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {play_res_x}\nPlayResY: {play_res_y}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{cfg.SUBTITLE_FONT},{cfg.SUBTITLE_FONT_SIZE},"
        f"{cfg.SUBTITLE_COLOR},{cfg.SUBTITLE_COLOR},{cfg.SUBTITLE_OUTLINE_COLOR},{cfg.SUBTITLE_SHADOW_COLOR},"
        # Bold field = 1 → libass renders the bold face of SUBTITLE_FONT (do NOT rename the font).
        f"{1 if cfg.SUBTITLE_BOLD else 0},0,0,0,100,100,0,0,1,{cfg.SUBTITLE_OUTLINE_SIZE},{cfg.SUBTITLE_SHADOW_SIZE},{alignment},10,10,{cfg.SUBTITLE_BOTTOM_MARGIN},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [
        f"Dialogue: 0,{format_ass_time(s['start'])},{format_ass_time(s['end'])},"
        f"Default,,0,0,0,,{_caption_text(s['text']).replace(chr(10), chr(92)+'N')}"
        for s in segments
    ]
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))
    log.info(f"✅ ASS subtitles: {ass_path}")
    return ass_path


# ─────────────────────────────────────────────
# Step 7 — Burn Subtitles into Video
# ─────────────────────────────────────────────
def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Return the real (width, height) of a video via ffprobe.

    ffprobe with -of csv=p=0:s=x can emit a trailing separator
    (e.g. "1080x1920x"), so we filter to numeric tokens rather than
    relying on an exact split.
    """
    out = run(
        f'ffprobe -v error -select_streams v:0 '
        f'-show_entries stream=width,height -of csv=p=0:s=x "{video_path}"',
        "Probe video dimensions",
    )
    first_line = next((l for l in out.splitlines() if l.strip()), "")
    nums = [tok for tok in first_line.strip().split("x") if tok.strip().isdigit()]
    if len(nums) < 2:
        raise RuntimeError(
            f"Could not parse video dimensions from ffprobe output: {out!r}"
        )
    return int(nums[0]), int(nums[1])


def _escape_filter_value(value: str) -> str:
    """Backslash-escape a value for use as an ffmpeg filtergraph option value.

    Filter options use `:` as a separator and `\\` / `'` as escape chars; `[],;` are
    special at the filterchain level. We backslash-escape all of them (NOT single-quote
    wrapping — that breaks the option-name parser when a second option follows). Spaces
    are not special to the filtergraph parser, so they pass through untouched.
    """
    s = str(value).replace("\\", "\\\\")
    for ch in ":'[],;":
        s = s.replace(ch, "\\" + ch)
    return s


def burn_subtitles(video_path: Path, segments: list[dict], output_dir: Path) -> Path:
    output_path = output_dir / "video_subtitled.mp4"
    log.info("🎬 Burning subtitles into video...")

    # Decide the working video. With 1080p conversion enabled we burn onto the converted
    # file; otherwise we burn directly onto the input. Either way `working_path` is always
    # bound, so CONVERT_TO_1080P=false no longer raises NameError.
    if cfg.CONVERT_TO_1080P:
        working_path = output_dir / "video_h264.mp4"
        run(
            f'ffmpeg -y -i "{video_path}" -c:v libx264 -crf 18 -preset fast '
            f'-vf "scale=1080:1920,format=yuv420p" -c:a aac "{working_path}"',
            "Convert to h264 1080p",
        )
    else:
        working_path = Path(video_path)

    # ASS PlayResX/Y must match the real frame, or positioning and scaling drift.
    play_res_x, play_res_y = get_video_dimensions(working_path)
    ass_path = generate_ass(segments, output_dir, play_res_x, play_res_y)

    # Build the filter with EXPLICIT option names and NO single quotes — ffmpeg's parser
    # rejects `subtitles='a':fontsdir='b'` (the quoted blob after `:` confuses option-name
    # parsing). filename= is a controlled basename (ffmpeg runs with cwd=output_dir);
    # fontsdir pins the typeface so it can't silently change via fontconfig fallback.
    # Produces, e.g.:
    #   subtitles=filename=subtitles.ass:fontsdir=/System/Library/Fonts/Supplemental
    #   subtitles=filename=subtitles.ass:fontsdir=/Users/x/My Fonts   (space is fine, no shell)
    sub_filter = f"subtitles=filename={_escape_filter_value(ass_path.name)}"
    font_dir = Path(cfg.SUBTITLE_FONT_PATH).parent
    if font_dir.is_dir():
        sub_filter += f":fontsdir={_escape_filter_value(font_dir)}"

    log.info("🎨 Burning ASS subtitles in a single ffmpeg pass...")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(working_path),
            "-vf", sub_filter,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            output_path.name,
        ],
        cwd=str(output_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error(f"ffmpeg subtitles error: {result.stderr}")
        raise RuntimeError("Failed to burn subtitles")

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

    out = resolve_output_dir(output_dir, allow_existing=steps is not None)
    log.info(f"🚀 Starting pipeline for: {video_path.name}")
    log.info(f"📁 Output directory: {out}")

    all_steps = ["audio", "enhance", "merge", "transcribe", "remove_fillers", "fix", "subtitles", "format", "speed", "metadata"]
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
        video_cleaned   = out / "video_cleaned.mp4"
        video_enhanced  = out / "video_enhanced.mp4"

        if video_subtitled.exists() and "subtitles" not in active:
            state["video"] = video_subtitled
            log.info(f"📂 Using existing subtitled video: {video_subtitled}")
        elif video_cleaned.exists() and "remove_fillers" not in active:
            state["video"] = video_cleaned
            log.info(f"📂 Using filler-cleaned video: {video_cleaned}")
        elif video_enhanced.exists():
            state["video"] = video_enhanced
            log.info(f"📂 Using enhanced video: {video_enhanced}")
        else:
            dest = out / video_path.name
            if not dest.exists():
                shutil.copy2(video_path, dest)
            state["video"] = dest
            log.info(f"📂 Using original video: {dest}")

    # Load existing word-level data if present. Preference: filler-cleaned words
    # (when not re-running filler removal) > hand-edited > raw.
    words_cleaned = out / "words_cleaned.json"
    words_edit    = out / "words_edit.json"
    words_raw     = out / "words_raw.json"
    if "transcribe" not in active:
        if words_cleaned.exists() and "remove_fillers" not in active:
            with open(words_cleaned, encoding="utf-8") as f:
                state["words"] = json.load(f)
            log.info(f"📂 Loaded filler-cleaned words: {words_cleaned}")
        elif words_edit.exists():
            with open(words_edit, encoding="utf-8") as f:
                state["words"] = json.load(f)
            log.info(f"📂 Loaded edited words: {words_edit}")
        elif words_raw.exists():
            with open(words_raw, encoding="utf-8") as f:
                state["words"] = json.load(f)
            log.info(f"📂 Loaded raw words: {words_raw}")

    # 4. Transcription — saves raw + editable words, does NOT group.
    if "transcribe" in active:
        audio_src = state.get("enhanced_audio") or state.get("audio")
        state["words"] = transcribe(audio_src, out)

    # 4b. Filler removal — cut слова-паразити from video + words BEFORE grouping.
    if "remove_fillers" in active and "words" in state:
        state["video"], state["words"] = remove_fillers(state["words"], state["video"], out)

    # 5. (Decoupled) LLM fix is intentionally OUT of the timed path — it desyncs
    #    text and timing. Hand-edit words_edit.json instead. fix_transcript stays
    #    opt-in via `--steps fix` only and is not part of the subtitle/burn path.

    # 6. Subtitles — group edited words into captions, then style + burn.
    if "subtitles" in active and "words" in state:
        state["segments"] = group_words_into_captions(state["words"])
        generate_srt(state["segments"], out)  # SRT kept as a side artifact
        state["video"] = burn_subtitles(state["video"], state["segments"], out)

    # 7. Video formatting
    if "format" in active:
        state["video"] = format_video(state["video"], out)

    # 8. Speed
    if "speed" in active:
        state["video"] = speed_up_video(state["video"], out)

    # 9. Metadata
    if "metadata" in active:
        # Derive segments from words if grouping didn't run this invocation.
        if "segments" not in state and "words" in state:
            state["segments"] = group_words_into_captions(state["words"])
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
        choices=["audio", "enhance", "merge", "transcribe", "remove_fillers", "fix", "subtitles", "format", "speed", "metadata"],
        help="Run only specified steps (default: all)",
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.steps)


if __name__ == "__main__":
    main()