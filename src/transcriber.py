"""Transcription engine wrapping faster-whisper.

A single WhisperModel is loaded once per process and reused across all files.
Each audio chunk is transcribed independently; the caller is responsible for
providing the correct time_offset so that returned Segment timestamps are
absolute (relative to the start of the original file).
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Tuple

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

    logger.info(
        "Loading model '%s' on %s with compute_type=%s",
        config.model,
        config.device,
        config.compute_type,
    )
    return WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
        num_workers=1,      # intra-model parallelism handled by ctranslate2
        cpu_threads=4,
    )


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
        condition_on_previous_text=True,
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
