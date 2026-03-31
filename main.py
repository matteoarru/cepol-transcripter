#!/usr/bin/env python3
"""Entry point — orchestrates scanning, transcription and output writing.

Usage
-----
    python main.py /path/to/media/root [options]

The pipeline:
  1. Scan root recursively for pending media files.
  2. Load the Whisper model once.
  3. For each file:
       a. Determine total duration via ffprobe.
       b. Stream audio chunks through ffmpeg (never loads the full file).
       c. Transcribe each chunk with faster-whisper on GPU.
       d. Write merged .txt and .srt output alongside the source file.
  4. Print a summary.
"""

import logging
import sys

from src.config import load_env
load_env()  # populate os.environ from .env before any other import reads it
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, Tuple

from tqdm import tqdm

from src.audio import audio_chunks, get_duration, is_video, prepare_audio
from src.cli import parse_args
from src.scanner import count_media_files, iter_media_files
from src.transcriber import TranscriptionResult, load_model, transcribe_file
from src.writer import write_srt, write_txt


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    """Configure root logger with a structured format.

    Args:
        level: Logging level string (e.g. ``"INFO"``).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Single-file processing (runs in the main process or a worker process)
# ---------------------------------------------------------------------------

def process_file(media_path: Path, config, model=None) -> dict:
    """Transcribe one media file and write its outputs.

    This function is self-contained so it can be called from either the main
    process (workers=1) or a :class:`~concurrent.futures.ProcessPoolExecutor`
    worker (workers > 1).  When *model* is ``None`` a new model is loaded
    inside the function (worker-process case).

    Args:
        media_path: Path to the media file to transcribe.
        config:     Runtime :class:`~src.config.TranscriptionConfig`.
        model:      Pre-loaded WhisperModel (optional, main-process only).

    Returns:
        A status dict with keys ``path``, ``ok``, ``elapsed``, ``error``.
    """
    logger = logging.getLogger(__name__)
    t0 = time.monotonic()

    if model is None:
        model = load_model(config)

    try:
        duration = get_duration(media_path)
        logger.info(
            "START: %s  (duration=%.1fs / %.1fmin)",
            media_path.name,
            duration,
            duration / 60,
        )

        short_name = media_path.name[:35]
        chunk_bar = tqdm(
            total=int(duration),
            desc=f"  {short_name} [preparing]",
            unit="s",
            bar_format=(
                "{l_bar}{bar}| {percentage:3.0f}% "
                "{n:.0f}/{total:.0f}s "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
            leave=False,
            dynamic_ncols=True,
        )

        with chunk_bar:
            # ── Phase 1: prepare audio (extract + cache for video files) ──────
            if is_video(media_path):
                chunk_bar.set_description_str(f"  {short_name} [extracting]")
            audio_source, was_cached = prepare_audio(media_path)
            if was_cached:
                logger.info("Reusing cached audio: %s", audio_source.name)

            # ── Phase 2: chunk + transcribe ───────────────────────────────────
            with audio_chunks(audio_source, config.chunk_size, config.max_duration) as chunk_iter:
                result: TranscriptionResult = transcribe_file(
                    model=model,
                    media_path=media_path,
                    duration=duration,
                    chunk_offsets_and_paths=_with_progress(
                        chunk_iter, chunk_bar, short_name, config.chunk_size, duration
                    ),
                    config=config,
                    on_chunk_done=lambda idx, elapsed: logger.debug(
                        "    chunk %d done in %.1fs", idx, elapsed
                    ),
                )

            # ── Phase 3: write outputs ────────────────────────────────────────
            chunk_bar.set_description_str(f"  {short_name} [writing]")
            txt_path = media_path.with_suffix(".txt")
            srt_path = media_path.with_suffix(".srt")
            write_txt(result.segments, txt_path)
            write_srt(result.segments, srt_path)

        elapsed = time.monotonic() - t0
        rtf = elapsed / duration if duration > 0 else 0
        logger.info(
            "DONE:  %s  [%.1fs wall | RTF=%.2f | %d segments]",
            media_path.name,
            elapsed,
            rtf,
            len(result.segments),
        )
        return {"path": str(media_path), "ok": True, "elapsed": elapsed, "error": None}

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("FAIL:  %s  — %s", media_path.name, exc, exc_info=True)
        return {"path": str(media_path), "ok": False, "elapsed": elapsed, "error": str(exc)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(root: Path, config, log_level: str = "INFO") -> None:
    """Main orchestration loop.

    Scans *root*, loads the model, and processes all pending media files.
    When ``config.workers > 1`` each worker loads its own model instance
    (suitable for multi-GPU setups).

    Args:
        root:      Root directory to scan.
        config:    Runtime configuration.
        log_level: Logging verbosity string.
    """
    _configure_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("Scanning: %s", root)
    pending = list(iter_media_files(root))
    total_files = len(pending)

    if total_files == 0:
        logger.info("No pending media files found. Exiting.")
        return

    logger.info("Found %d file(s) to process.", total_files)

    results: list[dict] = []
    wall_start = time.monotonic()

    if config.workers == 1:
        # Single-process: load model once, process sequentially.
        model = load_model(config)
        global_bar = tqdm(pending, total=total_files, desc="Files", unit="file", dynamic_ncols=True)
        for media_path in global_bar:
            global_bar.set_postfix_str(media_path.name[:40])
            result = process_file(media_path, config, model=model)
            results.append(result)
    else:
        # Multi-process: each worker loads its own model.
        logger.warning(
            "workers=%d — each worker loads its own model. "
            "Only use with multiple GPUs or CPU device.",
            config.workers,
        )
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(process_file, p, config): p
                for p in pending
            }
            global_bar = tqdm(
                as_completed(futures),
                total=total_files,
                desc="Files",
                unit="file",
                dynamic_ncols=True,
            )
            for future in global_bar:
                result = future.result()
                results.append(result)
                global_bar.set_postfix_str(Path(result["path"]).name[:40])

    _print_summary(results, time.monotonic() - wall_start)


def _with_progress(
    chunk_iter,
    bar: tqdm,
    short_name: str,
    chunk_size: int,
    total_duration: float,
) -> Generator[Tuple[Path, float], None, None]:
    """Wrap a chunk iterator so the tqdm bar advances by audio seconds per chunk
    and displays the current phase (extracting / transcribing).

    Phase labelling:
    * **extracting** — shown while ffmpeg is producing the next chunk WAV.
    * **transcribing** — shown while faster-whisper processes the chunk.

    Execution resumes after each ``yield`` only when ``transcribe_file``
    requests the next chunk, so the bar updates exactly once per chunk
    immediately after that chunk has been transcribed.

    Args:
        chunk_iter:     Iterator of ``(chunk_path, offset)`` from audio_chunks.
        bar:            tqdm instance with ``total`` set to file duration (s).
        short_name:     Truncated filename for the bar description.
        chunk_size:     Configured maximum chunk length in seconds.
        total_duration: Total file duration in seconds (used to cap last chunk).
    """
    bar.set_description_str(f"  {short_name} [extracting]")
    for chunk_path, offset in chunk_iter:
        # ffmpeg has finished producing this chunk — whisper is next
        bar.set_description_str(f"  {short_name} [transcribing]")
        yield chunk_path, offset
        # whisper finished — advance bar, then ffmpeg starts for the next chunk
        processed = min(float(chunk_size), total_duration - offset)
        bar.update(int(processed))
        bar.set_description_str(f"  {short_name} [extracting]")


def _print_summary(results: list[dict], total_elapsed: float) -> None:
    """Log a structured final summary to stdout.

    Args:
        results:       List of per-file result dicts from :func:`process_file`.
        total_elapsed: Total wall-clock time for the entire run.
    """
    logger = logging.getLogger(__name__)
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Total files : %d", len(results))
    logger.info("  Succeeded   : %d", len(ok))
    logger.info("  Failed      : %d", len(failed))
    logger.info("  Wall time   : %.1fs (%.1fmin)", total_elapsed, total_elapsed / 60)

    if failed:
        logger.warning("Failed files:")
        for r in failed:
            logger.warning("  [FAIL] %s — %s", r["path"], r["error"])

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point — parse arguments and run the pipeline."""
    args, config = parse_args()

    if not args.root.exists():
        print(f"ERROR: Root path does not exist: {args.root}", file=sys.stderr)
        sys.exit(1)

    run(args.root, config, log_level=args.log_level)


if __name__ == "__main__":
    main()
