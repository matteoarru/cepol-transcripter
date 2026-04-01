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
import gc
from dataclasses import replace
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.cli import parse_args
from src.config import TranscriptionConfig
from src.processor import (
    FileProcessingResult,
    initialize_worker_runtime,
    process_file,
    process_file_in_worker,
)
from src.scanner import iter_media_files
from src.transcriber import (
    detect_visible_cuda_devices,
    load_model,
    missing_cuda_runtime_libraries,
    warmup_model,
)


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
    visible_cuda_devices = detect_visible_cuda_devices(config)
    worker_config, model = _prepare_runtime(
        config,
        preload_model=config.workers == 1,
        visible_cuda_devices=visible_cuda_devices,
    )
    effective_workers = _effective_worker_count(worker_config, visible_cuda_devices)
    if effective_workers == 1 and model is None:
        model = load_model(worker_config)

    logger.info(
        "Runtime: device=%s compute_type=%s workers=%d cpu_threads=%d "
        "pipeline_threads=%d beam_size=%d best_of=%d condition_on_previous_text=%s",
        worker_config.device,
        worker_config.compute_type,
        effective_workers,
        worker_config.cpu_threads,
        worker_config.pipeline_threads,
        worker_config.beam_size,
        worker_config.best_of,
        worker_config.condition_on_previous_text,
    )

    if effective_workers == 1:
        # Single-process: load model once, process sequentially.
        global_bar = tqdm(pending, total=total_files, desc="Files", unit="file", dynamic_ncols=True)
        for media_path in global_bar:
            global_bar.set_postfix_str(media_path.name[:40])
            result = process_file(media_path, worker_config, model=model)
            results.append(result)
    else:
        # Multi-process: each worker loads its model once and reuses it.
        worker_runtime_config = _worker_runtime_config(worker_config, effective_workers)
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=initialize_worker_runtime,
            initargs=(worker_runtime_config, visible_cuda_devices),
        ) as executor:
            futures = {
                executor.submit(process_file_in_worker, p): p
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
        total_model_load = sum(r.metrics.model_load_elapsed for r in ok if r.metrics)
        total_probe = sum(r.metrics.probe_elapsed for r in ok if r.metrics)
        total_prepare = sum(r.metrics.prepare_elapsed for r in ok if r.metrics)
        total_extract = sum(r.metrics.extract_elapsed for r in ok if r.metrics)
        total_transcribe = sum(r.metrics.transcribe_elapsed for r in ok if r.metrics)
        total_write = sum(r.metrics.write_elapsed for r in ok if r.metrics)
        total_chunks = sum(r.metrics.chunks for r in ok if r.metrics)

        logger.info(
            "  Stage totals: model_load=%.1fs | probe=%.1fs | prepare=%.1fs | "
            "extract=%.1fs | transcribe=%.1fs | write=%.1fs | chunks=%d",
            total_model_load,
            total_probe,
            total_prepare,
            total_extract,
            total_transcribe,
            total_write,
            total_chunks,
        )
        logger.info(
            "  Note: extract and transcribe totals can overlap in pipelined runs."
        )

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


def _prepare_runtime(
    config: TranscriptionConfig,
    *,
    preload_model: bool,
    visible_cuda_devices: tuple[int, ...],
):
    """Run one startup device probe and return the effective runtime config."""
    logger = logging.getLogger(__name__)
    config = _normalise_runtime_config(config)

    if config.device != "cuda":
        model = load_model(config) if preload_model else None
        return config, model

    if not visible_cuda_devices:
        logger.warning("CUDA requested but no visible CUDA devices were detected. Falling back to CPU.")
        return _cpu_fallback_runtime(config, preload_model)

    missing_libs = missing_cuda_runtime_libraries()
    if missing_libs:
        logger.warning(
            "CUDA requested but required runtime libraries are missing: %s. Falling back to CPU.",
            ", ".join(missing_libs),
        )
        return _cpu_fallback_runtime(config, preload_model)

    load_started_at = time.monotonic()
    model = load_model(config)
    load_elapsed = time.monotonic() - load_started_at
    warmup_started_at = time.monotonic()

    try:
        warmup_model(model, config)
    except Exception as exc:
        if _is_cuda_runtime_failure(exc):
            logger.warning(
                "CUDA startup probe failed (%s). Falling back to CPU.",
                exc,
            )
            return _cpu_fallback_runtime(config, preload_model)
        raise

    warmup_elapsed = time.monotonic() - warmup_started_at
    logger.info(
        "CUDA startup probe succeeded: visible_devices=%d model_load=%.2fs warmup=%.2fs",
        len(visible_cuda_devices),
        load_elapsed,
        warmup_elapsed,
    )

    if preload_model:
        return config, model

    del model
    gc.collect()
    return config, None


def _cpu_fallback_runtime(config: TranscriptionConfig, preload_model: bool):
    """Return a CPU-safe runtime configuration and optional preloaded model."""
    fallback_config = replace(config, device="cpu", compute_type="int8", device_index=0)
    model = load_model(fallback_config) if preload_model else None
    return fallback_config, model


def _normalise_runtime_config(config: TranscriptionConfig) -> TranscriptionConfig:
    """Return a runtime-safe configuration for the selected backend."""
    if config.device != "cpu":
        return config

    if config.compute_type in {"int8", "float32"}:
        return config

    logging.getLogger(__name__).warning(
        "CPU mode does not support compute_type=%s efficiently. Switching to int8.",
        config.compute_type,
    )
    return replace(config, compute_type="int8", device_index=0)


def _effective_worker_count(
    config: TranscriptionConfig,
    visible_cuda_devices: tuple[int, ...],
) -> int:
    """Return the effective worker count after device-aware clamping."""
    if config.workers <= 1:
        return 1

    if config.device != "cuda":
        return config.workers

    available_gpus = len(visible_cuda_devices)
    if available_gpus <= 1:
        logging.getLogger(__name__).warning(
            "workers=%d requested on a single visible GPU. Using 1 worker for best throughput.",
            config.workers,
        )
        return 1

    if config.workers > available_gpus:
        logging.getLogger(__name__).warning(
            "workers=%d requested but only %d visible GPU(s). Capping workers to %d.",
            config.workers,
            available_gpus,
            available_gpus,
        )
    return min(config.workers, available_gpus)


def _worker_runtime_config(
    config: TranscriptionConfig,
    workers: int,
) -> TranscriptionConfig:
    """Return the per-worker configuration used in multi-process mode."""
    if config.device != "cpu" or workers <= 1:
        return config

    per_worker_threads = max(1, config.cpu_threads // workers)
    if per_worker_threads == config.cpu_threads:
        return config

    logging.getLogger(__name__).info(
        "CPU mode with %d workers: reducing cpu_threads per worker from %d to %d.",
        workers,
        config.cpu_threads,
        per_worker_threads,
    )
    return replace(config, cpu_threads=per_worker_threads)


def _is_cuda_runtime_failure(exc: Exception) -> bool:
    """Return True when an exception looks like a broken CUDA runtime."""
    message = str(exc).lower()
    retry_tokens = (
        "cuda",
        "cublas",
        "cudnn",
        "libcublas",
        "libcudnn",
        "ctranslate2",
    )
    return any(token in message for token in retry_tokens)


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
