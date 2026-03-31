#!/usr/bin/env python3
"""Entry point — orchestrates scanning, transcription and output writing.

Usage
-----
    python main.py /path/to/media/root [options]

The pipeline:
  1. Scan root recursively for pending media files.
  2. Load the Whisper model once per process.
  3. Delegate per-file processing to ``src.processor``.
  4. Print a summary with output locations.
"""

import logging
import sys

from src.config import load_env
load_env()  # populate os.environ from .env before any other import reads it
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.cli import parse_args
from src.processor import FileProcessingResult, process_file
from src.scanner import iter_media_files
from src.transcriber import load_model


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

    results: list[FileProcessingResult] = []
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
                global_bar.set_postfix_str(result.media_path.name[:40])

    _print_summary(results, time.monotonic() - wall_start)


def _print_summary(results: list[FileProcessingResult], total_elapsed: float) -> None:
    """Log a structured final summary to stdout.

    Args:
        results:       List of per-file results from :func:`process_file`.
        total_elapsed: Total wall-clock time for the entire run.
    """
    logger = logging.getLogger(__name__)
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Total files : %d", len(results))
    logger.info("  Succeeded   : %d", len(ok))
    logger.info("  Failed      : %d", len(failed))
    logger.info("  Wall time   : %.1fs (%.1fmin)", total_elapsed, total_elapsed / 60)

    if ok:
        logger.info("Saved outputs:")
        for r in ok:
            logger.info("  [OK]  %s", r.media_path)
            logger.info("        TXT: %s", r.txt_path)
            logger.info("        SRT: %s", r.srt_path)

    if failed:
        logger.warning("Failed files:")
        for r in failed:
            logger.warning("  [FAIL] %s — %s", r.media_path, r.error)

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
