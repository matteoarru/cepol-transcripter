"""Transcription engine wrapping faster-whisper.

A single WhisperModel is loaded once per process and reused across all files.
Each audio chunk is transcribed independently; the caller is responsible for
providing the correct time_offset so that returned Segment timestamps are
absolute (relative to the start of the original file).
"""

import ctypes
import importlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Tuple
import wave

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

from .config import (
    BEAM_SIZE,
    BEST_OF,
    COMPRESSION_RATIO_THRESHOLD,
    LOG_PROB_THRESHOLD,
    NO_SPEECH_THRESHOLD,
    TEMPERATURE,
    VAD_PARAMETERS,
    TranscriptionConfig,
)

logger = logging.getLogger(__name__)
_CUDA_RUNTIME_BOOTSTRAPPED = False


CUDA_RUNTIME_LIBRARY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("libcuda.so.1",),
    ("libcublas.so.12",),
    (
        "libcudnn.so.9",
        "libcudnn_ops_infer.so.9",
        "libcudnn_cnn_infer.so.9",
    ),
)


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A single transcription segment with absolute timestamps.

    Attributes:
        start: Start time in seconds (relative to original file start).
        end:   End time in seconds (relative to original file start).
        text:  Transcribed text with original casing and punctuation.
    """

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Aggregated result for a complete media file.

    Attributes:
        segments:       Ordered list of transcription segments.
        duration:       Total duration of the original file in seconds.
        language:       Detected (or forced) language code.
        elapsed:        Wall-clock time taken to transcribe (seconds).
    """

    segments: list[Segment] = field(default_factory=list)
    duration: float = 0.0
    language: str = "en"
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def load_model(config: TranscriptionConfig) -> "WhisperModel":
    """Instantiate and return a :class:`~faster_whisper.WhisperModel`.

    The model is loaded onto the device specified in *config*.  This is an
    expensive one-time operation; callers should cache the returned object.

    Args:
        config: Runtime transcription configuration.

    Returns:
        Loaded :class:`~faster_whisper.WhisperModel` instance.
    """
    from faster_whisper import WhisperModel  # noqa: PLC0415  lazy import
    _bootstrap_pip_cuda_runtime()

    logger.info(
        "Loading model '%s' on %s with compute_type=%s",
        config.model,
        config.device,
        config.compute_type,
    )
    return WhisperModel(
        config.model,
        device=config.device,
        device_index=config.device_index,
        compute_type=config.compute_type,
        num_workers=1,      # intra-model parallelism handled by ctranslate2
        cpu_threads=config.cpu_threads,
    )


def detect_visible_cuda_devices(config: TranscriptionConfig) -> tuple[int, ...]:
    """Return the CUDA device indexes visible to the current process."""
    if config.visible_gpu_ids:
        return tuple(index for index, _ in enumerate(config.visible_gpu_ids))

    try:
        import ctranslate2  # noqa: PLC0415  lazy import

        return tuple(range(ctranslate2.get_cuda_device_count()))
    except Exception:
        return ()


def missing_cuda_runtime_libraries() -> tuple[str, ...]:
    """Return missing CUDA userspace libraries required by faster-whisper."""
    _bootstrap_pip_cuda_runtime()
    missing: list[str] = []

    for candidates in CUDA_RUNTIME_LIBRARY_GROUPS:
        if any(_can_load_shared_library(candidate) for candidate in candidates):
            continue
        missing.append(candidates[0])

    return tuple(missing)


def warmup_model(model: "WhisperModel", config: TranscriptionConfig) -> None:
    """Run a tiny silent transcription to surface broken CUDA runtimes early."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
        with wave.open(tmp_file.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            wav_file.writeframes(b"\x00\x00" * 16_000)

        segments, _ = model.transcribe(
            tmp_file.name,
            language=config.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            suppress_blank=True,
            without_timestamps=True,
        )
        list(segments)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_chunk(
    model: "WhisperModel",
    audio_path: Path,
    time_offset: float,
    config: TranscriptionConfig,
) -> list[Segment]:
    """Transcribe one audio chunk and return absolute-timestamp segments.

    The *time_offset* is added to every segment's start/end so that the
    returned segments carry timestamps relative to the beginning of the
    original (possibly much longer) file.

    Args:
        model:       Loaded WhisperModel instance.
        audio_path:  Path to a WAV file containing the chunk audio.
        time_offset: Start position of this chunk in the original file (s).
        config:      Runtime configuration.

    Returns:
        List of :class:`Segment` objects with absolute timestamps.
    """
    vad_params = VAD_PARAMETERS if config.vad_filter else None

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=config.language,
        beam_size=config.beam_size,
        best_of=config.best_of,
        temperature=config.temperature,
        vad_filter=config.vad_filter,
        vad_parameters=vad_params,
        word_timestamps=False,
        condition_on_previous_text=config.condition_on_previous_text,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        # Law enforcement: preserve casing & punctuation
        suppress_blank=True,
        without_timestamps=False,
    )

    segments: list[Segment] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(
            Segment(
                start=seg.start + time_offset,
                end=seg.end + time_offset,
                text=text,
            )
        )

    logger.debug(
        "Chunk @%.1fs → %d segments (lang=%s, prob=%.2f)",
        time_offset,
        len(segments),
        info.language,
        info.language_probability,
    )
    return segments


def transcribe_file(
    model: "WhisperModel",
    media_path: Path,
    duration: float,
    chunk_offsets_and_paths: Iterable[Tuple[Path, float]],
    config: TranscriptionConfig,
    on_chunk_done: Optional[Callable[[int, float], None]] = None,
) -> TranscriptionResult:
    """Transcribe all chunks of a media file and merge into one result.

    This function is chunk-agnostic: it receives an iterable of
    ``(chunk_path, offset)`` pairs (produced by :func:`audio.audio_chunks`)
    and merges the per-chunk segments into a single ordered list.

    Args:
        model:                   Loaded WhisperModel instance.
        media_path:              Original media file (for logging only).
        duration:                Total file duration in seconds.
        chunk_offsets_and_paths: Iterable of (wav_path, start_offset) tuples.
        config:                  Runtime configuration.
        on_chunk_done:           Optional callback(chunk_index, elapsed_ms).

    Returns:
        :class:`TranscriptionResult` with merged segments.
    """
    t_start = time.monotonic()
    all_segments: list[Segment] = []
    detected_language = config.language

    for chunk_index, (chunk_path, offset) in enumerate(chunk_offsets_and_paths):
        chunk_start = time.monotonic()
        logger.info(
            "  Chunk %d | offset=%.1fs | file=%s",
            chunk_index,
            offset,
            media_path.name,
        )

        chunk_segments = transcribe_chunk(model, chunk_path, offset, config)
        all_segments.extend(chunk_segments)

        chunk_elapsed = time.monotonic() - chunk_start
        if on_chunk_done:
            on_chunk_done(chunk_index, chunk_elapsed)

    elapsed = time.monotonic() - t_start
    return TranscriptionResult(
        segments=all_segments,
        duration=duration,
        language=detected_language,
        elapsed=elapsed,
    )


def _can_load_shared_library(name: str) -> bool:
    """Return True when *name* can be loaded through the dynamic linker."""
    try:
        ctypes.CDLL(name)
        return True
    except OSError:
        return False


def _bootstrap_pip_cuda_runtime() -> None:
    """Preload NVIDIA runtime libraries when they are installed via pip."""
    global _CUDA_RUNTIME_BOOTSTRAPPED

    if _CUDA_RUNTIME_BOOTSTRAPPED:
        return

    runtime_dirs: list[Path] = []
    for module_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        module_paths = list(getattr(module, "__path__", []))
        if module_paths:
            runtime_dirs.append(Path(module_paths[0]).resolve())
            continue

        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        runtime_dirs.append(Path(module_file).resolve().parent)

    if not runtime_dirs:
        _CUDA_RUNTIME_BOOTSTRAPPED = True
        return

    existing_paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    merged_paths = [str(path) for path in runtime_dirs]
    for path in existing_paths:
        if path not in merged_paths:
            merged_paths.append(path)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged_paths)

    for runtime_dir in runtime_dirs:
        for pattern in ("libcublas*.so*", "libcudnn*.so*"):
            for library_path in sorted(runtime_dir.glob(pattern)):
                try:
                    ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue

    logger.info(
        "CUDA runtime bootstrap: using Python-packaged libraries from %s",
        ", ".join(str(path) for path in runtime_dirs),
    )
    _CUDA_RUNTIME_BOOTSTRAPPED = True
