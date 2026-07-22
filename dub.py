#!/usr/bin/env python3
"""
Automated Video Dubbing System
===============================
IdeaLabs Digital — Internship Assignment

Turns a YouTube video in any language into an English-dubbed version:
same video, same pacing, English audio.

Pipeline
--------
1. Fetch      : download the source video with yt-dlp
2. Transcribe : extract speech segments (with timestamps) using Whisper
3. Translate  : carry each segment's meaning into natural English
4. Synthesize : generate natural English speech per segment with edge-tts
5. Time-fit   : stretch/compress each clip so it lands back on its original
                timestamp, so the dub stays in sync with the video
6. Remix      : swap the new audio track into the original video (video
                stream is copied, not re-encoded, so quality/length is untouched)

Usage
-----
    python dub.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" \\
        --output out/dubbed.mp4 \\
        --whisper-model medium \\
        --voice en-US-GuyNeural

Resuming a crashed/interrupted run
-----------------------------------
    python dub.py <same url and flags as before> --resume "work\\job_1784..."

Run `python dub.py --help` for all options.
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    """Timestamped progress print so long jobs are legible in the terminal."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, raising with captured output on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n--- stderr ---\n{result.stderr}"
        )
    return result


def check_dependencies() -> None:
    for tool in ("ffmpeg", "yt-dlp"):
        if shutil.which(tool) is None:
            sys.exit(f"Required tool '{tool}' not found on PATH. Please install it first.")


@dataclass
class Segment:
    start: float   # seconds, in the source timeline
    end: float
    text: str      # English text (already translated)


# --------------------------------------------------------------------------- #
# Step 1 — Fetch
# --------------------------------------------------------------------------- #

def download_video(url: str, workdir: Path, cookies_from_browser: str | None = None,
                    cookies_file: str | None = None) -> Path:
    # Resume support: if a source video is already sitting in this job's
    # workdir from a previous (interrupted) run, don't re-download it.
    existing = sorted(workdir.glob("source.*"))
    existing = [c for c in existing if c.suffix.lower() in (".mp4", ".mkv", ".webm") and c.stat().st_size > 0]
    if existing:
        log(f"Step 1/4: Source video already present, skipping download -> {existing[0].name}")
        return existing[0]

    log("Step 1/4: Downloading source video (yt-dlp)...")
    out_template = str(workdir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--remote-components", "ejs:github",  # let yt-dlp fetch its JS challenge solver (Deno alone isn't enough)
        "-o", out_template,
    ]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    elif cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd.append(url)
    run(cmd)
    candidates = sorted(workdir.glob("source.*"))
    video_files = [c for c in candidates if c.suffix.lower() in (".mp4", ".mkv", ".webm")]
    if not video_files:
        raise RuntimeError("yt-dlp did not produce a video file.")
    video_path = video_files[0]
    log(f"  downloaded -> {video_path.name}")
    return video_path


def extract_audio(video_path: Path, workdir: Path) -> Path:
    """Pull a mono 16kHz wav out for Whisper (its expected input format)."""
    audio_path = workdir / "audio.wav"
    if audio_path.exists() and audio_path.stat().st_size > 0:
        log("  audio already extracted, skipping -> audio.wav")
        return audio_path
    run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)])
    return audio_path


# --------------------------------------------------------------------------- #
# Step 2+3 — Transcribe & Translate
# --------------------------------------------------------------------------- #
#
# Design decision: faster-whisper supports task="translate", which transcribes
# non-English speech AND translates it to English in one pass, using Whisper's
# own multilingual training rather than a literal word-for-word pipeline. This
# keeps meaning-preserving, natural phrasing (the assignment's ask) without
# needing a separate translation model/API per language. Segment-level
# timestamps come out of the same call, which we need later for sync.
#
# If you want higher translation quality for Indian languages specifically,
# swap this step for: Whisper (task="transcribe", original language) ->
# IndicTrans2 (translate). See README for how to plug that in.

def _get_duration_seconds(path: Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(result.stdout.strip())


def transcribe_and_translate(audio_path: Path, model_size: str, compute_type: str = "int8",
                              use_vad: bool = True, chunk_seconds: int = 600) -> list[Segment]:
    workdir = audio_path.parent
    cache_path = workdir / "segments.json"

    # Resume support: transcription (especially with medium/large models on
    # a 2hr file) is expensive. If we already have a cached result from a
    # previous run of this same job, reuse it instead of redoing it.
    if cache_path.exists():
        log("Step 2/4: Using cached transcription (segments.json found)...")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        segments = [Segment(**s) for s in cached]
        log(f"  {len(segments)} speech segments loaded from cache")
        return segments

    from faster_whisper import WhisperModel

    log(f"Step 2/4: Transcribing + translating speech to English (Whisper '{model_size}', compute_type='{compute_type}')...")
    model = WhisperModel(model_size, device="auto", compute_type=compute_type)

    # Long audio (an hour+) fed to faster-whisper in one shot can blow up memory:
    # it builds the full spectrogram for the whole file before decoding starts,
    # which for a ~2hr file needs several GB in a single allocation. Splitting
    # into fixed-length chunks first keeps peak memory bounded regardless of
    # total video length, at the cost of a few extra ffmpeg calls.
    total_duration = _get_duration_seconds(audio_path)
    segments: list[Segment] = []
    offset = 0.0
    chunk_idx = 0

    # Checkpoint file: written after EVERY chunk (not just at the very end),
    # so interrupting the run (Ctrl+C, closed laptop, crash) only loses the
    # one chunk in progress, not all transcription done so far. On restart,
    # already-checkpointed chunks are skipped entirely.
    checkpoint_path = workdir / "segments.partial.json"
    resume_offset = 0.0
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        segments = [Segment(**s) for s in checkpoint["segments"]]
        resume_offset = checkpoint["completed_offset"]
        chunk_idx = checkpoint["chunk_idx"]
        offset = resume_offset
        log(f"  resuming transcription from {offset:.0f}s "
            f"({len(segments)} segments already done in a previous run)...")

    log(f"  running language detection + decoding in {chunk_seconds}s chunks (this can take a bit on "
        "CPU for larger models, you should see per-segment lines appear shortly)...")

    detected_language = None
    detected_prob = None

    while offset < total_duration:
        chunk_path = workdir / f"audio_chunk_{chunk_idx}.wav"
        run([
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(offset), "-t", str(chunk_seconds),
            "-ac", "1", "-ar", "16000", str(chunk_path),
        ])

        segments_iter, info = model.transcribe(
            str(chunk_path),
            task="translate",       # -> English text directly
            vad_filter=use_vad,     # skip silence, improves segment boundaries (can misfire on singing/music)
            beam_size=5,
        )
        if detected_language is None:
            detected_language, detected_prob = info.language, info.language_probability

        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            start, end = seg.start + offset, seg.end + offset
            segments.append(Segment(start=start, end=end, text=text))
            log(f"  [{start:7.1f}s -> {end:7.1f}s] {text}")

        chunk_path.unlink(missing_ok=True)
        offset += chunk_seconds
        chunk_idx += 1

        # Checkpoint after every chunk completes.
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "segments": [asdict(s) for s in segments],
                "completed_offset": offset,
                "chunk_idx": chunk_idx,
            }, f)

    if detected_language is not None:
        log(f"  detected source language: {detected_language} (confidence {detected_prob:.2f})")
    log(f"  {len(segments)} speech segments found")

    # Cache the final result under the name synthesize/resume expects, and
    # drop the chunk-level checkpoint now that we have a complete result.
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in segments], f)
    checkpoint_path.unlink(missing_ok=True)

    return segments


# --------------------------------------------------------------------------- #
# Step 4a — Synthesize English speech per segment
# --------------------------------------------------------------------------- #

# Backoff schedule for transient edge-tts failures (dropped connections,
# rate limits, etc). 5 retries after the first attempt, 6 attempts total.
_TTS_RETRY_WAITS = [2, 4, 8, 16, 30]


async def _synth_one(text: str, voice: str, out_path: Path) -> None:
    import edge_tts

    last_err = None
    for attempt in range(len(_TTS_RETRY_WAITS) + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(out_path))
            # edge-tts occasionally "succeeds" but writes an empty/near-empty
            # file on transient errors — treat that as a failure and retry.
            if out_path.exists() and out_path.stat().st_size > 0:
                return
            last_err = RuntimeError("edge-tts produced an empty file")
        except Exception as e:
            last_err = e

        if attempt < len(_TTS_RETRY_WAITS):
            wait = _TTS_RETRY_WAITS[attempt]
            log(f"    TTS call failed ({last_err}); retrying in {wait}s "
                f"(attempt {attempt + 2}/{len(_TTS_RETRY_WAITS) + 1})...")
            await asyncio.sleep(wait)

    raise RuntimeError(f"edge-tts failed after {len(_TTS_RETRY_WAITS) + 1} attempts: {last_err}")


def synthesize_segments(segments: list[Segment], workdir: Path, voice: str) -> list[Path]:
    log(f"Step 3/4: Synthesizing English speech per segment (voice={voice})...")
    clip_dir = workdir / "clips"
    clip_dir.mkdir(exist_ok=True)
    paths = []
    skipped = 0
    for i, seg in enumerate(segments):
        out_path = clip_dir / f"seg_{i:04d}.mp3"
        # Resume support: skip clips already synthesized (and non-empty) from
        # a previous run of this same job.
        if out_path.exists() and out_path.stat().st_size > 0:
            paths.append(out_path)
            skipped += 1
            continue
        asyncio.run(_synth_one(seg.text, voice, out_path))
        paths.append(out_path)
        log(f"  synthesized {i + 1}/{len(segments)}")
    if skipped:
        log(f"  ({skipped} clips reused from a previous run)")
    return paths


# --------------------------------------------------------------------------- #
# Step 4b — Time-fit each clip to its original slot, then assemble full track
# --------------------------------------------------------------------------- #

def get_duration(path: Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(result.stdout.strip())


def time_fit_clip(src: Path, dst: Path, target_duration: float) -> None:
    """
    Stretch or compress a synthesized clip so it fits the original segment's
    duration, keeping the dub in sync with mouth movements / scene timing.
    Clamped to a sane range so speech doesn't become garbled or absurdly slow.
    """
    if dst.exists() and dst.stat().st_size > 0:
        return  # resume support: already fitted in a previous run

    src_duration = get_duration(src)
    if src_duration <= 0 or target_duration <= 0:
        shutil.copy(src, dst)
        return

    tempo = src_duration / target_duration
    tempo = max(0.5, min(2.0, tempo))  # ffmpeg atempo's single-filter safe range

    run([
        "ffmpeg", "-y", "-i", str(src),
        "-filter:a", f"atempo={tempo:.4f}",
        str(dst),
    ])


def assemble_audio_track(
    segments: list[Segment],
    raw_clips: list[Path],
    workdir: Path,
    total_duration: float,
) -> Path:
    # NOTE: an earlier version built one giant ffmpeg command with an -i flag
    # and adelay filter per segment. That works for a handful of segments,
    # but on a longer/denser video (hundreds+ of segments), the resulting
    # command line exceeds Windows' command-line length limit and
    # subprocess.run fails with "The filename or extension is too long".
    #
    # A pydub-overlay loop fixes that, but doesn't scale to a 2-hour video:
    # AudioSegment.overlay() re-processes the *entire* track on every call,
    # so mixing gets progressively slower as the track fills up — with
    # thousands of segments over 2 hours this could take a very long time.
    #
    # Instead: mix directly into a numpy int32 accumulation buffer. Each
    # clip is only touched once, for its own length — so total work is
    # proportional to total audio content, not to (segments × track length).
    import numpy as np
    from pydub import AudioSegment

    log("Step 4/4: Fitting clips to timing and assembling final audio track...")
    fitted_dir = workdir / "fitted"
    fitted_dir.mkdir(exist_ok=True)

    fitted_clips = []
    for i, (seg, clip) in enumerate(zip(segments, raw_clips)):
        target = max(seg.end - seg.start, 0.05)
        fitted_path = fitted_dir / f"fit_{i:04d}.mp3"
        time_fit_clip(clip, fitted_path, target)
        fitted_clips.append((seg.start, fitted_path))
        log(f"  fitted {i + 1}/{len(segments)}")

    log("  mixing all segments onto the timeline...")
    sample_rate = 24000
    total_samples = int(total_duration * sample_rate) + sample_rate  # +1s padding
    buffer = np.zeros(total_samples, dtype=np.int32)

    for i, (start, path) in enumerate(fitted_clips):
        clip_audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(sample_rate)
        clip_samples = np.array(clip_audio.get_array_of_samples(), dtype=np.int32)

        start_sample = int(start * sample_rate)
        end_sample = start_sample + len(clip_samples)

        # Clip to buffer bounds (a clip stretched near the very end of the
        # video could otherwise run past our allocated array).
        if end_sample > total_samples:
            clip_samples = clip_samples[: total_samples - start_sample]
            end_sample = total_samples
        if start_sample < total_samples and len(clip_samples) > 0:
            buffer[start_sample:end_sample] += clip_samples

        if (i + 1) % 50 == 0 or (i + 1) == len(fitted_clips):
            log(f"  mixed {i + 1}/{len(fitted_clips)}")

    # Clip back down to int16 range in case overlapping segments summed too loud.
    buffer = np.clip(buffer, -32768, 32767).astype(np.int16)

    final_audio = workdir / "dubbed_audio.wav"
    mixed = AudioSegment(
        buffer.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
    )
    mixed.export(str(final_audio), format="wav")
    log(f"  assembled -> {final_audio.name}")
    return final_audio


# --------------------------------------------------------------------------- #
# Step 5 — Remix into final video
# --------------------------------------------------------------------------- #

def remux(video_path: Path, dubbed_audio: Path, output_path: Path) -> None:
    log("Remixing: swapping in dubbed audio (video re-encode skipped)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(dubbed_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ])
    log(f"Done -> {output_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Dub a YouTube video into English.")
    parser.add_argument("url", nargs="?", help="YouTube URL to dub")
    parser.add_argument("--output", "-o", default="output/dubbed.mp4", help="Output file path")
    parser.add_argument("--whisper-model", default="medium",
                         choices=["tiny", "base", "small", "medium", "large-v3"],
                         help="Whisper model size (bigger = better quality, slower). Default: medium")
    parser.add_argument("--compute-type", default="int8",
                         choices=["int8", "int8_float16", "float16", "float32"],
                         help="faster-whisper compute precision. int8 is fastest on CPU (default). "
                              "Use float16 if you have a CUDA GPU.")
    parser.add_argument("--no-vad", action="store_true",
                         help="Disable voice-activity-detection filtering. Try this if 0 segments are "
                              "found on content like singing/music, where VAD can misfire.")
    parser.add_argument("--cookies-from-browser", default=None,
                         help="Browser to pull YouTube cookies from (chrome, edge, firefox, brave, etc.) "
                              "Use this if yt-dlp reports 'Sign in to confirm you're not a bot'.")
    parser.add_argument("--cookies-file", default=None,
                         help="Path to a Netscape-format cookies.txt file (e.g. exported via a browser "
                              "extension). Takes priority over --cookies-from-browser if both are given; "
                              "often more reliable on Windows where browser cookie DBs are locked/encrypted.")
    parser.add_argument("--voice", default="en-US-GuyNeural",
                         help="edge-tts voice name, e.g. en-US-GuyNeural, en-US-JennyNeural, "
                              "en-GB-RyanNeural. Run `edge-tts --list-voices` to see all options.")
    parser.add_argument("--keep-temp", action="store_true",
                         help="Keep the working directory (downloaded video, clips, etc.) for inspection "
                              "even on success.")
    parser.add_argument("--resume", default=None, metavar="WORKDIR",
                         help="Path to a previous job's work directory (printed when a run fails) to "
                              "continue from — reuses the downloaded video, extracted audio, cached "
                              "transcription, and any clips already synthesized.")
    args = parser.parse_args()

    url = args.url or input("YouTube URL: ").strip()
    if not url:
        sys.exit("No URL provided.")

    check_dependencies()

    output_path = Path(args.output)
    if args.resume:
        workdir = Path(args.resume)
        if not workdir.exists():
            sys.exit(f"--resume path does not exist: {workdir}")
        log(f"Resuming job from: {workdir}")
    else:
        workdir = Path("work") / f"job_{int(time.time())}"
        workdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        video_path = download_video(url, workdir, args.cookies_from_browser, args.cookies_file)
        audio_path = extract_audio(video_path, workdir)
        total_duration = get_duration(video_path)

        segments = transcribe_and_translate(audio_path, args.whisper_model, args.compute_type,
                                             use_vad=not args.no_vad)
        if not segments:
            sys.exit("No speech detected in this video; nothing to dub.")

        raw_clips = synthesize_segments(segments, workdir, args.voice)
        dubbed_audio = assemble_audio_track(segments, raw_clips, workdir, total_duration)
        remux(video_path, dubbed_audio, output_path)

        elapsed = time.time() - t0
        log(f"Finished in {elapsed / 60:.1f} minutes. Output: {output_path.resolve()}")
    except (Exception, SystemExit):
        # Never delete progress on failure — that's what turned one dropped
        # TTS connection into 3 lost hours before. Keep everything so
        # --resume can pick up right where it left off.
        log(f"Run failed/interrupted. Job files kept at: {workdir}")
        log(f'Rerun with: --resume "{workdir}" (plus your usual url and other flags) to continue.')
        raise
    else:
        if not args.keep_temp:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()