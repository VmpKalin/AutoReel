# 🎬 ClipForge — Automated Video Processing Pipeline

Fully automated video processing pipeline: audio enhancement → transcription → subtitles → formatting → social media metadata.

---

## ✨ Features

- 🔊 **Audio Enhancement** — Noise reduction and loudness normalization via Auphonic API
- 🎙️ **Accurate Transcription** — WhisperX with word-level timestamps for frame-perfect subtitles
- ✍️ **AI Text Correction** — Local Ollama model fixes grammar and punctuation without sending data to the cloud
- 📝 **Burned-in Subtitles** — Auto-scaled, styled subtitles rendered directly into the video via MoviePy
- 📐 **Format Conversion** — Convert to any aspect ratio: 9:16, 16:9, 1:1, 4:5
- 📱 **Social Media Metadata** — Claude AI generates optimized titles, captions and hashtags for Instagram & TikTok
- 🔇 **Watermark Removal** — Automatically detects and removes Auphonic free-tier audio watermark

---

## 🛠️ Requirements

- Python 3.10+
- ffmpeg
- Ollama (for local AI text correction)

```bash
# macOS
brew install ffmpeg
brew install ollama

# Ubuntu/Debian
sudo apt install ffmpeg
```

---

## ⚡ Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/yourname/clipforge.git
cd clipforge

python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Pull local AI model

```bash
brew services start ollama      # run Ollama in background on Mac
ollama pull llama3.1:8b         # ~5GB download
```

### 3. Configure

```bash
cp .env.example .env
# Fill in your API keys
```

### 4. Run

```bash
# Full pipeline
python3 pipeline.py my_video.mov

# From a specific step onwards
python3 pipeline.py my_video.mov --steps fix subtitles format metadata

# Custom output directory
python3 pipeline.py my_video.mov -o ./results
```

---

## 📋 Pipeline Steps

| Step | What it does |
|------|-------------|
| `audio` | Extracts audio track via ffmpeg |
| `enhance` | Enhances audio via Auphonic API (denoising, normalization) |
| `remove_watermark` | Detects and removes Auphonic free-tier watermark from audio |
| `merge` | Replaces original audio in video with enhanced version |
| `transcribe` | Transcribes with WhisperX using word-level timestamps |
| `fix` | Corrects grammar and punctuation via local Ollama model |
| `subtitles` | Generates SRT + ASS files and burns them into the video |
| `format` | Converts video to target aspect ratio (9:16, 16:9, etc.) |
| `metadata` | Generates Instagram and TikTok captions and hashtags via Claude API |

---

## ⚙️ Configuration (.env)

```env
# ─── Auphonic ────────────────────────────────────
AUPHONIC_API_KEY=your_auphonic_api_key
REMOVE_AUPHONIC_WATERMARK=true

# ─── Anthropic Claude ────────────────────────────
ANTHROPIC_API_KEY=your_anthropic_api_key

# ─── WhisperX ────────────────────────────────────
WHISPER_MODEL=large-v3     # tiny / base / small / medium / large-v3
WHISPER_DEVICE=cpu         # cpu / cuda
WHISPER_LANGUAGE=en        # en / uk / ru / de ...

# ─── Subtitles ───────────────────────────────────
SUBTITLE_FONT_SIZE=60
SUBTITLE_OUTLINE_SIZE=3
SUBTITLE_POSITION=bottom   # bottom / top / center

# ─── Video Format ────────────────────────────────
OUTPUT_FORMAT=9:16         # 9:16 / 16:9 / 1:1 / 4:5 / original
ADD_PADDING=true           # true = black bars, false = crop
CONVERT_TO_1080P=true      # convert HEVC/4K to h264 1080p for processing
```

---

## 📁 Output Structure

```
output/
├── audio_original.wav        # Extracted audio
├── audio_enhanced.wav        # Enhanced audio (Auphonic)
├── audio_trimmed.wav         # Audio with watermark removed
├── video_enhanced.mp4        # Video with enhanced audio
├── video_h264.mp4            # h264 converted for subtitle rendering
├── transcript_raw.json       # Raw transcript with timestamps (JSON)
├── transcript.txt            # Raw transcript with timestamps (readable)
├── transcript_fixed.json     # Corrected transcript (JSON)
├── transcript_fixed.txt      # Corrected transcript (readable)
├── subtitles.srt             # SRT subtitle file
├── subtitles.ass             # ASS subtitle file (styled)
├── video_subtitled.mp4       # Video with burned-in subtitles
├── video_formatted.mp4       # Final formatted video
├── metadata.json             # Title, captions, hashtags
└── pipeline.log              # Execution log
```

---

## 🔑 API Keys

| Service | Where to get | Cost |
|---------|-------------|------|
| **Auphonic** | auphonic.com → Account → API Access | 2 hrs/month free |
| **Anthropic** | console.anthropic.com → API Keys | ~$0.01 per video |

> Ollama runs 100% locally — free and private

---

## ✂️ Bonus: Video Trimmer

```bash
# Trim first 15 seconds
python3 trim.py my_video.mov 15

# From second 5 to second 20
python3 trim.py my_video.mov 15 --start 5

# Custom output file
python3 trim.py my_video.mov 30 -o short_clip.mp4
```

---

## 🐛 Troubleshooting

**venv not activated:**
```bash
source venv/bin/activate
# You should see (venv) at the start of the terminal line
```

**Ollama not running:**
```bash
brew services start ollama
```

**WhisperX too slow on CPU:**
```env
WHISPER_MODEL=base    # much faster, slightly lower quality
```

**Font not found (subtitle crosses):**
```bash
find /System/Library/Fonts -name "*.ttf" | grep -i arial
# Then set the full path in burn_subtitles → font=
```

**ffmpeg not found:**
```bash
brew install ffmpeg
```
