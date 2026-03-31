"""Single-file transcription workflow.

This module owns the full lifecycle for one media file:
1. Probe duration and decide the chunking strategy.
2. Stream chunks through ffmpeg.
3. Transcribe them with faster-whisper.
4. Write sibling `.txt` and `.srt` outputs.
"""

from contextlib import ExitStack
from dataclasses import dataclass, replace
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Optional, Sequence

from tqdm import tqdm

from .audio import (
    audio_chunks_from_plan,
    build_chunk_plan,
    get_duration,
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


@dataclass(frozen=True)
class MediaProcessingPlan:
    """Precomputed processing decisions for one media file."""

    duration: float
    processed_duration: float
    chunk_plan: Sequence["ChunkSpec"]
    use_pipeline: bool


@dataclass(frozen=True)
class FileProcessingResult:
    """Outcome of processing one media file."""

    media_path: Path
    ok: bool
    elapsed: float
    error: str | None = None
    txt_path: Path | None = None
    srt_path: Path | None = None


def process_file(
    media_path: Path,
    config: TranscriptionConfig,
    model: Optional["WhisperModel"] = None,
) -> FileProcessingResult:
    """Transcribe one media file and write sibling outputs."""
    started_at = time.monotonic()

    try:
        if model is None:
            model = load_model(config)

        transcription, txt_path, srt_path = _process_file_once(
            media_path,
            config,
            model,
        )

        elapsed = time.monotonic() - started_at
        _log_processing_complete(
            media_path,
            transcription,
            elapsed,
            transcription.duration,
        )
        return FileProcessingResult(
            media_path=media_path,
            ok=True,
            elapsed=elapsed,
            txt_path=txt_path,
            srt_path=srt_path,
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
                transcription, txt_path, srt_path = _process_file_once(
                    media_path,
                    fallback_config,
                    load_model(fallback_config),
                )
                elapsed = time.monotonic() - started_at
                _log_processing_complete(
                    media_path,
                    transcription,
                    elapsed,
                    transcription.duration,
                )
                return FileProcessingResult(
                    media_path=media_path,
                    ok=True,
                    elapsed=elapsed,
                    txt_path=txt_path,
                    srt_path=srt_path,
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
        )


def _process_file_once(
    media_path: Path,
    config: TranscriptionConfig,
    model: "WhisperModel",
) -> tuple[TranscriptionResult, Path, Path]:
    """Process one media file with a single, already-loaded model."""
    plan = _build_processing_plan(media_path, config)
    _log_processing_start(media_path, plan)

    with _chunk_progress_bar(media_path.name, plan.processed_duration) as chunk_bar:
        transcription = _transcribe_media(media_path, model, plan, config, chunk_bar)
        txt_path, srt_path = _write_outputs(media_path, transcription)

    return transcription, txt_path, srt_path


def _build_processing_plan(
    media_path: Path,
    config: TranscriptionConfig,
) -> MediaProcessingPlan:
    """Return a single source of truth for per-file processing decisions."""
    duration = get_duration(media_path)
    processed_duration = _effective_duration(duration, config.max_duration)
    return MediaProcessingPlan(
        duration=duration,
        processed_duration=processed_duration,
        chunk_plan=build_chunk_plan(processed_duration, config.chunk_size),
        use_pipeline=_should_pipeline_media(processed_duration, config),
    )


def _log_processing_start(media_path: Path, plan: MediaProcessingPlan) -> None:
    """Log the start of a media file with its effective processing scope."""
    logger.info(
        "START: %s  (duration=%.1fs / %.1fmin | processed=%.1fs)",
        media_path.name,
        plan.duration,
        plan.duration / 60,
        plan.processed_duration,
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
) -> TranscriptionResult:
    """Choose the chunk source strategy and transcribe the file."""
    short_name = media_path.name[:35]

    with ExitStack() as stack:
        if plan.use_pipeline:
            logger.info(
                "Using pipelined chunk extraction: chunk_size=%ds, pipeline_threads=%d",
                config.chunk_size,
                config.pipeline_threads,
            )
            chunk_iter = stack.enter_context(
                pipelined_audio_chunks_from_plan(
                    media_path,
                    plan.chunk_plan,
                    max_prefetch=config.pipeline_threads,
                )
            )
        else:
            chunk_source = _prepare_chunk_source(media_path, chunk_bar)
            chunk_iter = stack.enter_context(
                audio_chunks_from_plan(chunk_source, plan.chunk_plan)
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
            on_chunk_done=lambda idx, elapsed: logger.debug(
                "    chunk %d done in %.1fs", idx, elapsed
            ),
        )


def _prepare_chunk_source(media_path: Path, chunk_bar: tqdm) -> Path:
    """Return the source path used by the sequential chunking strategy."""
    if is_video(media_path):
        chunk_bar.set_description_str(f"  {media_path.name[:35]} [extracting]")

    audio_source, was_cached = prepare_audio(media_path)
    if was_cached:
        logger.info("Reusing cached audio: %s", audio_source.name)
    return audio_source


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


def _log_processing_complete(
    media_path: Path,
    transcription: TranscriptionResult,
    elapsed: float,
    processed_duration: float,
) -> None:
    """Log the final success message for a media file."""
    rtf = elapsed / processed_duration if processed_duration > 0 else 0.0
    logger.info(
        "DONE:  %s  [%.1fs wall | RTF=%.2f | %d segments]",
        media_path.name,
        elapsed,
        rtf,
        len(transcription.segments),
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
    return replace(config, device="cpu", compute_type="int8")
