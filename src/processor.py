"""Single-file transcription workflow.

This module owns the full lifecycle for one media file:
1. Probe duration and choose the chunk/source strategy.
2. Stream chunks through ffmpeg.
3. Transcribe them with faster-whisper.
4. Write sibling `.txt` and `.srt` outputs.
"""

from contextlib import ExitStack
from dataclasses import dataclass, replace
import logging
from multiprocessing import current_process
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Generator, Optional, Sequence

from tqdm import tqdm

from .audio import (
    cached_audio_path,
    audio_chunks_from_plan,
    build_chunk_plan,
    get_duration,
    has_reusable_audio_cache,
    is_video,
    pipelined_audio_chunks_from_plan,
    prepare_audio,
)
from .config import TranscriptionConfig
from .output_paths import transcript_output_paths
from .transcriber import TranscriptionResult, load_model, transcribe_file
from .writer import write_srt, write_txt

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from .audio import ChunkSpec


logger = logging.getLogger(__name__)

_WORKER_MODEL: Optional["WhisperModel"] = None
_WORKER_CONFIG: Optional[TranscriptionConfig] = None


@dataclass(frozen=True)
class MediaProcessingPlan:
    """Precomputed processing decisions for one media file."""

    duration: float
    processed_duration: float
    chunk_plan: Sequence["ChunkSpec"]
    use_pipeline: bool
    source_strategy: str


@dataclass
class PerformanceMetrics:
    """Performance timings captured while processing one media file."""

    model_load_elapsed: float = 0.0
    probe_elapsed: float = 0.0
    prepare_elapsed: float = 0.0
    extract_elapsed: float = 0.0
    transcribe_elapsed: float = 0.0
    write_elapsed: float = 0.0
    chunks: int = 0


@dataclass(frozen=True)
class FileProcessingResult:
    """Outcome of processing one media file."""

    media_path: Path
    ok: bool
    elapsed: float
    error: str | None = None
    txt_path: Path | None = None
    srt_path: Path | None = None
    metrics: PerformanceMetrics | None = None
    source_strategy: str | None = None


def initialize_worker_runtime(
    config: TranscriptionConfig,
    cuda_device_indexes: tuple[int, ...] = (),
) -> None:
    """Load one model per worker process and pin it to a single GPU if needed."""
    global _WORKER_CONFIG, _WORKER_MODEL

    worker_config = config
    if config.device == "cuda" and cuda_device_indexes:
        slot = _worker_slot(len(cuda_device_indexes))
        worker_config = replace(config, device_index=cuda_device_indexes[slot])
        logger.info(
            "Worker %s pinned to CUDA device %d",
            current_process().name,
            worker_config.device_index,
        )

    _WORKER_CONFIG = worker_config
    _WORKER_MODEL = load_model(worker_config)


def process_file_in_worker(media_path: Path) -> FileProcessingResult:
    """Process a single file using the worker-local cached model."""
    if _WORKER_CONFIG is None or _WORKER_MODEL is None:
        raise RuntimeError("Worker runtime not initialised")
    return process_file(media_path, _WORKER_CONFIG, model=_WORKER_MODEL)


def process_file(
    media_path: Path,
    config: TranscriptionConfig,
    model: Optional["WhisperModel"] = None,
) -> FileProcessingResult:
    """Transcribe one media file and write sibling outputs."""
    started_at = time.monotonic()
    metrics = PerformanceMetrics()

    try:
        if model is None:
            load_started_at = time.monotonic()
            model = load_model(config)
            metrics.model_load_elapsed = time.monotonic() - load_started_at

        transcription, txt_path, srt_path, plan = _process_file_once(
            media_path,
            config,
            model,
            metrics,
        )

        elapsed = time.monotonic() - started_at
        _log_processing_complete(
            media_path,
            transcription,
            elapsed,
            transcription.duration,
            plan.source_strategy,
            metrics,
        )
        return FileProcessingResult(
            media_path=media_path,
            ok=True,
            elapsed=elapsed,
            txt_path=txt_path,
            srt_path=srt_path,
            metrics=metrics,
            source_strategy=plan.source_strategy,
        )

    except Exception as exc:
        if _should_retry_on_cpu(config, exc):
            fallback_config = _cpu_fallback_config(config)
            logger.warning(
                "CUDA transcription unavailable for %s (%s). Retrying on CPU with compute_type=%s.",
                media_path.name,
                exc,
                fallback_config.compute_type,
            )
            try:
                fallback_metrics = PerformanceMetrics()
                load_started_at = time.monotonic()
                fallback_model = load_model(fallback_config)
                fallback_metrics.model_load_elapsed = time.monotonic() - load_started_at
                transcription, txt_path, srt_path, plan = _process_file_once(
                    media_path,
                    fallback_config,
                    fallback_model,
                    fallback_metrics,
                )
                elapsed = time.monotonic() - started_at
                _log_processing_complete(
                    media_path,
                    transcription,
                    elapsed,
                    transcription.duration,
                    plan.source_strategy,
                    fallback_metrics,
                )
                return FileProcessingResult(
                    media_path=media_path,
                    ok=True,
                    elapsed=elapsed,
                    txt_path=txt_path,
                    srt_path=srt_path,
                    metrics=fallback_metrics,
                    source_strategy=plan.source_strategy,
                )
            except Exception as fallback_exc:
                exc = fallback_exc

        elapsed = time.monotonic() - started_at
        logger.error("FAIL:  %s  — %s", media_path.name, exc, exc_info=True)
        return FileProcessingResult(
            media_path=media_path,
            ok=False,
            elapsed=elapsed,
            error=str(exc),
            metrics=metrics,
        )


def _process_file_once(
    media_path: Path,
    config: TranscriptionConfig,
    model: "WhisperModel",
    metrics: PerformanceMetrics | None = None,
) -> tuple[TranscriptionResult, Path, Path, MediaProcessingPlan]:
    """Process one media file with a single, already-loaded model."""
    active_metrics = metrics or PerformanceMetrics()
    probe_started_at = time.monotonic()
    plan = _build_processing_plan(media_path, config)
    active_metrics.probe_elapsed += time.monotonic() - probe_started_at
    _log_processing_start(media_path, plan)

    with _chunk_progress_bar(media_path.name, plan.processed_duration) as chunk_bar:
        transcription = _transcribe_media(
            media_path,
            model,
            plan,
            config,
            chunk_bar,
            active_metrics,
        )
        write_started_at = time.monotonic()
        txt_path, srt_path = _write_outputs(media_path, transcription)
        active_metrics.write_elapsed += time.monotonic() - write_started_at
        _cleanup_generated_audio_cache(media_path)

    return transcription, txt_path, srt_path, plan


def _build_processing_plan(
    media_path: Path,
    config: TranscriptionConfig,
) -> MediaProcessingPlan:
    """Return a single source of truth for per-file processing decisions."""
    duration = get_duration(media_path)
    processed_duration = _effective_duration(duration, config.max_duration)
    use_pipeline = _should_pipeline_media(processed_duration, config)
    return MediaProcessingPlan(
        duration=duration,
        processed_duration=processed_duration,
        chunk_plan=build_chunk_plan(processed_duration, config.chunk_size),
        use_pipeline=use_pipeline,
        source_strategy=_select_source_strategy(media_path, use_pipeline),
    )


def _log_processing_start(media_path: Path, plan: MediaProcessingPlan) -> None:
    """Log the start of a media file with its effective processing scope."""
    logger.info(
        "START: %s  (duration=%.1fs / %.1fmin | processed=%.1fs | strategy=%s)",
        media_path.name,
        plan.duration,
        plan.duration / 60,
        plan.processed_duration,
        plan.source_strategy,
    )


def _chunk_progress_bar(filename: str, processed_duration: float) -> tqdm:
    """Return the per-file progress bar configured for chunk processing."""
    short_name = filename[:35]
    return tqdm(
        total=int(processed_duration),
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


def _transcribe_media(
    media_path: Path,
    model: "WhisperModel",
    plan: MediaProcessingPlan,
    config: TranscriptionConfig,
    chunk_bar: tqdm,
    metrics: PerformanceMetrics,
) -> TranscriptionResult:
    """Choose the chunk source strategy and transcribe the file."""
    short_name = media_path.name[:35]
    prepare_started_at = time.monotonic()
    chunk_source = _resolve_chunk_source(media_path, plan, chunk_bar)
    metrics.prepare_elapsed += time.monotonic() - prepare_started_at

    with ExitStack() as stack:
        if plan.use_pipeline:
            logger.info(
                "Using pipelined chunk extraction: chunk_size=%ds, pipeline_threads=%d",
                config.chunk_size,
                config.pipeline_threads,
            )
            chunk_iter = stack.enter_context(
                pipelined_audio_chunks_from_plan(
                    chunk_source,
                    plan.chunk_plan,
                    max_prefetch=config.pipeline_threads,
                    on_chunk_extracted=lambda spec, elapsed: _record_extraction(
                        metrics,
                        elapsed,
                    ),
                )
            )
        else:
            chunk_iter = stack.enter_context(
                audio_chunks_from_plan(
                    chunk_source,
                    plan.chunk_plan,
                    on_chunk_extracted=lambda spec, elapsed: _record_extraction(
                        metrics,
                        elapsed,
                    ),
                )
            )

        return transcribe_file(
            model=model,
            media_path=media_path,
            duration=plan.processed_duration,
            chunk_offsets_and_paths=_with_progress(
                chunk_iter,
                chunk_bar,
                short_name,
                config.chunk_size,
                plan.processed_duration,
            ),
            config=config,
            on_chunk_done=lambda idx, elapsed: _record_transcription(
                metrics,
                elapsed,
            ),
        )


def _resolve_chunk_source(
    media_path: Path,
    plan: MediaProcessingPlan,
    chunk_bar: tqdm,
) -> Path:
    """Return the audio source path chosen by the adaptive strategy."""
    if plan.source_strategy == "direct_media":
        return media_path

    if plan.source_strategy == "cached_audio":
        audio_source, _ = prepare_audio(media_path)
        logger.info("Using cached audio sidecar for pipelined chunks: %s", audio_source.name)
        return audio_source

    if plan.source_strategy == "prepared_audio":
        if is_video(media_path):
            chunk_bar.set_description_str(f"  {media_path.name[:35]} [extracting]")

        audio_source, was_cached = prepare_audio(media_path)
        if was_cached:
            logger.info("Reusing cached audio: %s", audio_source.name)
        return audio_source

    raise ValueError(f"Unknown source strategy: {plan.source_strategy}")


def _write_outputs(
    media_path: Path,
    transcription: TranscriptionResult,
) -> tuple[Path, Path]:
    """Write transcript outputs next to the source media file."""
    txt_path, srt_path = transcript_output_paths(media_path)
    write_txt(transcription.segments, txt_path)
    write_srt(transcription.segments, srt_path)
    logger.info("OUTPUT TXT: %s", txt_path)
    logger.info("OUTPUT SRT: %s", srt_path)
    return txt_path, srt_path


def _cleanup_generated_audio_cache(media_path: Path) -> None:
    """Remove the sibling ``*.audio.wav`` sidecar after a successful run."""
    if not is_video(media_path):
        return

    sidecar = cached_audio_path(media_path)
    if not sidecar.exists():
        return

    try:
        sidecar.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove audio cache %s: %s", sidecar, exc)
        return

    logger.info("CLEANUP AUDIO CACHE: %s", sidecar)


def _log_processing_complete(
    media_path: Path,
    transcription: TranscriptionResult,
    elapsed: float,
    processed_duration: float,
    source_strategy: str,
    metrics: PerformanceMetrics,
) -> None:
    """Log the final success message and timing breakdown for a media file."""
    rtf = elapsed / processed_duration if processed_duration > 0 else 0.0
    logger.info(
        "DONE:  %s  [%.1fs wall | RTF=%.2f | %d segments | strategy=%s]",
        media_path.name,
        elapsed,
        rtf,
        len(transcription.segments),
        source_strategy,
    )
    logger.info(
        "PERF:  %s  [model_load=%.2fs | probe=%.2fs | prepare=%.2fs | "
        "extract=%.2fs | transcribe=%.2fs | write=%.2fs | chunks=%d]",
        media_path.name,
        metrics.model_load_elapsed,
        metrics.probe_elapsed,
        metrics.prepare_elapsed,
        metrics.extract_elapsed,
        metrics.transcribe_elapsed,
        metrics.write_elapsed,
        metrics.chunks,
    )


def _with_progress(
    chunk_iter,
    bar: tqdm,
    short_name: str,
    chunk_size: int,
    total_duration: float,
) -> Generator[tuple[Path, float], None, None]:
    """Advance the chunk progress bar in sync with extraction and inference."""
    bar.set_description_str(f"  {short_name} [extracting]")
    for chunk_path, offset in chunk_iter:
        bar.set_description_str(f"  {short_name} [transcribing]")
        yield chunk_path, offset
        processed = min(float(chunk_size), total_duration - offset)
        bar.update(int(processed))
        bar.set_description_str(f"  {short_name} [extracting]")


def _effective_duration(duration: float, max_duration: float | None) -> float:
    """Return the actual amount of media that will be processed."""
    if max_duration is None:
        return duration
    return min(duration, max_duration)


def _should_pipeline_media(duration: float, config: TranscriptionConfig) -> bool:
    """Return True when chunk prefetching should be enabled."""
    return duration > float(config.chunk_size) and config.pipeline_threads > 1


def _select_source_strategy(media_path: Path, use_pipeline: bool) -> str:
    """Choose the extraction strategy based on media type and cache state."""
    if not use_pipeline:
        return "prepared_audio" if is_video(media_path) else "direct_media"

    if is_video(media_path) and has_reusable_audio_cache(media_path):
        return "cached_audio"

    return "direct_media"


def _record_extraction(metrics: PerformanceMetrics, elapsed: float) -> None:
    """Accumulate per-chunk extraction timing."""
    metrics.extract_elapsed += elapsed


def _record_transcription(metrics: PerformanceMetrics, elapsed: float) -> None:
    """Accumulate per-chunk transcription timing."""
    metrics.transcribe_elapsed += elapsed
    metrics.chunks += 1


def _worker_slot(total_slots: int) -> int:
    """Return a stable zero-based slot index for the current worker process."""
    if total_slots <= 1:
        return 0

    proc = current_process()
    identity = getattr(proc, "_identity", ())
    if identity:
        return (identity[0] - 1) % total_slots

    match = re.search(r"(\d+)$", proc.name)
    if match:
        return (int(match.group(1)) - 1) % total_slots

    return 0


def _should_retry_on_cpu(config: TranscriptionConfig, exc: Exception) -> bool:
    """Return True when a CUDA-related failure should trigger CPU retry."""
    if config.device != "cuda":
        return False

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


def _cpu_fallback_config(config: TranscriptionConfig) -> TranscriptionConfig:
    """Return a CPU-safe fallback configuration."""
    return replace(config, device="cpu", compute_type="int8", device_index=0)
