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
from typing import List

from tqdm import tqdm

from src.audio import audio_chunks, get_duration
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

        # Stream chunks through the context manager; each chunk is a temp WAV.
        with audio_chunks(media_path, config.chunk_size, config.max_duration) as chunk_iter:
            total_chunks = max(1, int(duration / config.chunk_size) + 1)

            chunk_bar = tqdm(
                chunk_iter,
                total=total_chunks,
                desc=f"  {media_path.name[:40]}",
                unit="chunk",
                leave=False,
                dynamic_ncols=True,
            )

            result: TranscriptionResult = transcribe_file(
                model=model,
                media_path=media_path,
                duration=duration,
                chunk_offsets_and_paths=chunk_bar,
                config=config,
                on_chunk_done=lambda idx, elapsed: logger.debug(
                    "    chunk %d done in %.1fs", idx, elapsed
                ),
            )

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

    results: List[dict] = []
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


def _print_summary(results: List[dict], total_elapsed: float) -> None:
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
