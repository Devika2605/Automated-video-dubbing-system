# Automated Video Dubbing System

Turns a YouTube video in any language into an English-dubbed version:
same video, same pacing, English audio.

## Setup

```bash
# System dependency (not pip-installable)
sudo apt install ffmpeg      # macOS: brew install ffmpeg

# Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
python dub.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" \
    --output output/dubbed.mp4 \
    --whisper-model medium \
    --voice en-US-GuyNeural
```

Options:
- `--whisper-model`: `tiny`/`base`/`small`/`medium`/`large-v3`. Bigger = more
  accurate but slower. `medium` is a good default; use `small` if you're
  CPU-only and want faster turnaround, `large-v3` if you have a GPU and want
  best translation quality.
- `--voice`: any [edge-tts voice](https://github.com/rany2/edge-tts) name.
  Run `edge-tts --list-voices` to browse options (different accents/genders).
- `--keep-temp`: keeps the working directory (`work/job_<timestamp>/`) around
  afterwards so you can inspect the downloaded video, per-segment audio clips,
  and intermediate files.

A GPU (CUDA) is picked up automatically by faster-whisper if available;
otherwise it runs on CPU (slower, especially for the 2-hour video).

## Architecture

```
YouTube URL
    │  yt-dlp
    ▼
source.mp4 ───────────────────────────────┐
    │  ffmpeg (extract audio)             │ (kept for final remux)
    ▼                                     │
audio.wav                                 │
    │  faster-whisper (task="translate")  │
    ▼                                     │
[Segment(start, end, english_text), ...]  │
    │  edge-tts, one call per segment     │
    ▼                                     │
raw English clips (variable length)       │
    │  ffmpeg atempo (stretch/compress    │
    │  each clip to its original slot)    │
    ▼                                     │
fitted clips, each = segment duration     │
    │  ffmpeg adelay + amix               │
    ▼                                     │
dubbed_audio.wav (full-length track)      │
    │ ffmpeg -map (swap audio, copy video)│
    ▼◄────────────────────────────────────┘
output/dubbed.mp4
```

### Key decisions

**Transcribe + translate in one step.** Whisper's `task="translate"` mode
transcribes non-English speech and translates it to English directly, using
the model's own multilingual training rather than a literal word-for-word
pass — which is what "translate for meaning" calls for. It also returns
segment-level timestamps for free, which the sync step needs. This avoids
wiring up a second translation model/API per source language.

*Trade-off:* for Indian languages specifically, a dedicated model like
**IndicTrans2** may produce more idiomatic English than Whisper's translate
mode. If translation quality on Hindi/Tamil/etc. content is a priority, swap
`transcribe_and_translate()` to: Whisper `task="transcribe"` (native
language) → IndicTrans2 (native → English), and keep everything else
unchanged (the `Segment` list format is the same either way).

**Per-segment timing fit, not a single global stretch.** Different segments
say different amounts in English vs. the original language, so a single
video-wide speed adjustment would drift out of sync. Instead each
synthesized clip is individually time-stretched (`ffmpeg atempo`, clamped to
0.5×–2× so speech doesn't degrade) to fill its original segment's slot, then
placed back at that segment's original start time with `adelay`. This is
what keeps the dub roughly lip-synced without needing video re-encoding for
frame-level A/V alignment.

**Video re-encoded never, audio re-encoded once.** The remux step uses
`-c:v copy` so the video stream is passed through untouched (same quality,
same length, fast). Only the audio is encoded, to AAC.

## Stretch goal: multi-speaker support

Not implemented in the base script, but here's how it plugs in if you want
to attempt it:

1. Run **speaker diarization** (`pyannote.audio`) on `audio.wav` to get
   `(start, end, speaker_id)` intervals.
2. When building `Segment`s, tag each with its `speaker_id` (intersect
   diarization intervals with Whisper's segment intervals).
3. Either map each `speaker_id` to a distinct `edge-tts` voice, or, for
   actual voice cloning, feed a clean audio sample of that speaker into
   **Coqui XTTS** instead of edge-tts for synthesis.
4. Everything downstream (time-fitting, mixing, remux) is unchanged — it
   just becomes "voice used per segment" as an extra parameter.

## Notes on runtime

- A 30-minute video: expect roughly 10-20 min end-to-end on a decent CPU with
  `--whisper-model medium` (transcription is the dominant cost); much faster
  with a GPU.
- A 2-hour video: scale accordingly — consider `small` model or a GPU if CPU
  time becomes impractical. Report your actual measured times per the
  assignment's submission requirements.
