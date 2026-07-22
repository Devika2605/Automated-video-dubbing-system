# Automated Video Dubbing System

Turns a YouTube video in any language into an English-dubbed version:
same video, same pacing, English audio.

## Setup

```bash
# System dependencies (not pip-installable)
sudo apt install ffmpeg      # macOS: brew install ffmpeg
# yt-dlp also needs a JS runtime to solve YouTube's challenge scripts —
# see "YouTube blocks the download" below if you hit that.

# Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
python dub.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" \
    --output out/dubbed.mp4 \
    --whisper-model medium \
    --voice en-US-GuyNeural
```

### Options

- `--whisper-model`: `tiny`/`base`/`small`/`medium`/`large-v3`. Bigger = more
  accurate but slower. `medium` is a good default; use `small` on a longer
  video (e.g. 2hrs+) if you're CPU-only and want a more practical runtime;
  `large-v3` if you have a GPU and want the best translation quality.
- `--compute-type`: `int8` (default, fastest on CPU) / `int8_float16` /
  `float16` (use this if you have a CUDA GPU) / `float32`.
- `--voice`: any [edge-tts voice](https://github.com/rany2/edge-tts) name.
  Run `edge-tts --list-voices` to browse options (different accents/genders).
- `--no-vad`: disables voice-activity-detection filtering. Use this if a run
  reports 0 speech segments found — Whisper's VAD is tuned for spoken
  speech and can filter out singing/music entirely.
- `--cookies-from-browser <browser>`: pulls YouTube auth cookies straight
  from a logged-in browser (`chrome`, `edge`, `firefox`, `brave`, etc). Use
  this if yt-dlp reports `Sign in to confirm you're not a bot`. Close the
  browser fully first — Windows locks the cookie database while it's running.
- `--cookies-file <path>`: points at an exported Netscape-format
  `cookies.txt` instead (e.g. via the "Get cookies.txt LOCALLY" browser
  extension). More reliable than `--cookies-from-browser` on Windows, where
  browser cookie stores are often locked or encrypted in a way yt-dlp can't
  read directly.
- `--keep-temp`: keeps the working directory (`work/job_<timestamp>/`)
  around after a **successful** run, so you can inspect the downloaded
  video, per-segment clips, and intermediate files. (On a failed/interrupted
  run, the working directory is always kept automatically — see Resuming
  below.)
- `--resume <workdir>`: continues a previous run from its job folder instead
  of starting over. See **Resuming an interrupted run** below.

A GPU (CUDA) is picked up automatically by faster-whisper if available;
otherwise it runs on CPU (slower, especially for the 2-hour video).

## Resuming an interrupted run

Long videos take a while, and things happen — a dropped connection, a closed
laptop, a Ctrl+C. This script is built to survive that:

- **On any failure**, the job's working directory (`work/job_<timestamp>/`)
  is **kept**, never deleted, and the script prints the exact command to
  continue:
  ```
  Rerun with: --resume "work/job_1784659147" (plus your usual url and other flags) to continue.
  ```
- **Just re-run your original command and add that `--resume` flag.** It
  will skip:
  - re-downloading the video, if `source.mp4` is already there
  - re-extracting audio, if `audio.wav` is already there
  - re-transcribing, using a **per-chunk checkpoint** — transcription is
    done in fixed-length chunks (10 minutes by default) and checkpointed
    after every single chunk, so an interruption only costs you at most one
    chunk's worth of re-work, not the whole transcript
  - re-synthesizing any segment whose audio clip was already generated

**Important:** this per-chunk checkpointing only applies going forward. A
job folder created by an older version of the script (before this
checkpointing existed) won't have partial transcript progress to resume
from — transcription for that job will have to restart from 0s, though the
downloaded video and extracted audio will still be reused.

edge-tts calls (the speech synthesis step) also retry automatically on
transient failures — up to 5 retries with increasing backoff (2s, 4s, 8s,
16s, 30s) — before giving up, so a single flaky network blip won't kill an
otherwise-fine run.

## Architecture

```
YouTube URL
    │  yt-dlp (+ cookies if needed)
    ▼
source.mp4 ────────────────────────────────────┐
    │  ffmpeg (extract audio)                  │ (kept for final remux)
    ▼                                           │
audio.wav                                       │
    │  split into fixed-length chunks           │
    │  (bounds memory use on long videos)       │
    ▼                                           │
audio_chunk_0.wav, audio_chunk_1.wav, ...       │
    │  faster-whisper (task="translate"),       │
    │  checkpointed after each chunk            │
    ▼                                           │
[Segment(start, end, english_text), ...]        │
    │  edge-tts, one call per segment,          │
    │  retried on transient failure              │
    ▼                                           │
raw English clips (variable length)             │
    │  ffmpeg atempo (stretch/compress          │
    │  each clip to its original slot)          │
    ▼                                           │
fitted clips, each = segment duration           │
    │  mixed in-memory (numpy accumulation      │
    │  buffer — see "Key decisions" below)      │
    ▼                                           │
dubbed_audio.wav (full-length track)            │
    │  ffmpeg -map (swap audio, copy video)     │
    ▼◄───────────────────────────────────────────┘
out/dubbed.mp4
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

**Audio is chunked before transcription.** Feeding an hour+ of audio to
faster-whisper in one shot builds the full spectrogram for the whole file
before decoding even starts, which for a ~2hr file needs several GB in a
single allocation. Splitting into fixed-length chunks (10 minutes by
default) up front keeps peak memory bounded regardless of total video
length, at the cost of a few extra ffmpeg calls — and, as a side benefit,
gives natural checkpoints for the resume system above.

**Per-segment timing fit, not a single global stretch.** Different segments
say different amounts in English vs. the original language, so a single
video-wide speed adjustment would drift out of sync. Instead each
synthesized clip is individually time-stretched (`ffmpeg atempo`, clamped to
0.5×–2× so speech doesn't degrade) to fill its original segment's slot, then
placed at that segment's original start time. This is what keeps the dub
roughly lip-synced without needing video re-encoding for frame-level A/V
alignment.

**Audio mixing happens in-memory (numpy), not via a single ffmpeg command.**
An earlier version built one ffmpeg command with an `-i` flag and `adelay`
filter per segment. That's fine for a handful of segments, but a longer or
denser video can easily produce 500+ segments — and the resulting command
line exceeds Windows' command-line length limit, causing the process launch
itself to fail. Mixing directly into a numpy accumulation buffer instead
touches each clip exactly once, for its own length, so total work scales
with total audio content rather than with (segment count × command-line
size), and there's no OS argument-length limit involved at all.

**Video re-encoded never, audio re-encoded once.** The remux step uses
`-c:v copy` so the video stream is passed through untouched (same quality,
same length, fast). Only the audio is encoded, to AAC.

**Progress is never silently discarded.** Downloads, audio extraction,
transcription (per-chunk), and synthesis (per-segment) are all checked for
already-completed output before redoing the work, and a failed run keeps
its working directory instead of deleting it — see **Resuming** above.

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

## Troubleshooting

**"Sign in to confirm you're not a bot"** — YouTube's bot detection. Use
`--cookies-from-browser <browser>` (close that browser fully first) or
`--cookies-file <path>` with an exported `cookies.txt` (more reliable on
Windows).

**"n challenge solving failed" / "Only images are available for
download"** — yt-dlp needs a JS runtime to solve YouTube's challenge script.
Install one (e.g. `winget install DenoLand.Deno` on Windows) and make sure
yt-dlp can reach its remote challenge-solver component (this script already
passes `--remote-components ejs:github` for you).

**0 speech segments found** — usually means the content is music/singing,
which Whisper's VAD often filters out as "not speech." Retry with `--no-vad`.

**A run crashes or is interrupted** — don't panic, don't delete anything.
The working directory is preserved automatically; look for the printed
`--resume "work/job_..."` command and re-run with that flag added. See
**Resuming an interrupted run** above.

## Notes on runtime

- A 30-minute video: expect roughly 10-20 min end-to-end on a decent CPU with
  `--whisper-model medium` (transcription is the dominant cost); much faster
  with a GPU.
- A 2-hour video: scale accordingly — consider `small` model or a GPU if CPU
  time becomes impractical. Report your actual measured times per the
  assignment's submission requirements.