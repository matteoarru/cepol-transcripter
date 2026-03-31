"""Audio extraction and chunking using ffmpeg subprocesses.

All heavy lifting is delegated to ffmpeg so we never load entire audio files
into Python memory — critical for 4-hour recordings.
"""

import json
import logging
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator, Tuple

from .config import (
    FFMPEG_CHANNELS,
    FFMPEG_CODEC,
    FFMPEG_SAMPLE_RATE,
    VIDEO_EXTENSIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_video(path: Path) -> bool:
    """Return True when *path* is a video file.

    Args:
        path: Media file path.

    Returns:
        True for video files; False for audio-only files.
    """
    return path.suffix.lower() in VIDEO_EXTENSIONS


def get_duration(path: Path) -> float:
    """Return the duration of a media file in seconds via ffprobe.

    Args:
        path: Any audio or video file.

    Returns:
        Duration in seconds as a float.

    Raises:
        RuntimeError: When ffprobe cannot read the file.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Cannot read duration from {path}: {exc}") from exc


def extract_audio(source: Path, target: Path) -> None:
    """Extract and normalise the audio track of *source* into *target*.

    Output is always 16 kHz, mono, 16-bit PCM WAV — the format Whisper
    expects.  For pure audio files the codec is simply transcoded; for video
    files the first audio stream is extracted.

    Args:
        source: Input media file (audio or video).
        target: Output WAV file path.

    Raises:
        RuntimeError: When ffmpeg returns a non-zero exit code.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-vn",                           # drop video stream
        "-acodec", FFMPEG_CODEC,
        "-ar", str(FFMPEG_SAMPLE_RATE),
        "-ac", str(FFMPEG_CHANNELS),
        str(target),
    ]
    _run_ffmpeg(cmd, source)


def extract_audio_segment(
    source: Path,
    target: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    """Extract a time-bounded segment of audio from *source* into *target*.

    Uses ffmpeg's fast seek (-ss before -i) so extraction time is O(segment)
    rather than O(file).

    Args:
        source: Input media file.
        target: Output WAV file for the segment.
        start_seconds: Segment start position in seconds.
        duration_seconds: Length of the segment in seconds.

    Raises:
        RuntimeError: When ffmpeg returns a non-zero exit code.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_seconds),       # fast seek BEFORE input
        "-i", str(source),
        "-t", str(duration_seconds),
        "-vn",
        "-acodec", FFMPEG_CODEC,
        "-ar", str(FFMPEG_SAMPLE_RATE),
        "-ac", str(FFMPEG_CHANNELS),
        str(target),
    ]
    _run_ffmpeg(cmd, source)


@contextmanager
def audio_chunks(
    source: Path,
    chunk_size: int,
    max_duration: float | None = None,
) -> Generator[Iterator[Tuple[Path, float]], None, None]:
    """Context manager that yields an iterator of (temp_wav_path, time_offset).

    The temporary directory holding chunk wav files is cleaned up automatically
    when the ``with`` block exits, ensuring no large files litter the disk.

    Usage::

        with audio_chunks(path, chunk_size=1800) as chunk_iter:
            for chunk_path, offset in chunk_iter:
                ...   # chunk_path is a valid WAV file here

    Args:
        source: Original media file (audio or video).
        chunk_size: Maximum chunk length in seconds.
        max_duration: Hard cap on total processed duration (optional).

    Yields:
        An iterator of ``(path_to_temp_wav, start_offset_seconds)`` tuples.
    """
    total_duration = get_duration(source)
    if max_duration is not None:
        total_duration = min(total_duration, max_duration)

    with tempfile.TemporaryDirectory(prefix="cepol_chunk_") as tmpdir:
        yield _chunk_iterator(source, Path(tmpdir), chunk_size, total_duration)


def _chunk_iterator(
    source: Path,
    tmpdir: Path,
    chunk_size: int,
    total_duration: float,
) -> Iterator[Tuple[Path, float]]:
    """Internal generator that materialises and yields chunks one at a time.

    Each chunk WAV is extracted by ffmpeg, yielded to the caller, then
    immediately deleted before the next chunk is extracted.  This keeps
    peak disk usage to a single chunk at any given moment.

    Args:
        source:         Original media file.
        tmpdir:         Temporary directory (managed by caller).
        chunk_size:     Maximum chunk length in seconds.
        total_duration: Duration to process (may be capped by max_duration).

    Yields:
        Tuples of (chunk_wav_path, absolute_start_offset_seconds).
    """
    offset = 0.0
    chunk_index = 0

    while offset < total_duration:
        remaining = total_duration - offset
        duration = min(float(chunk_size), remaining)
        chunk_path = tmpdir / f"chunk_{chunk_index:04d}.wav"

        logger.debug(
            "Extracting chunk %d: %.1fs – %.1fs from %s",
            chunk_index,
            offset,
            offset + duration,
            source.name,
        )
        extract_audio_segment(source, chunk_path, offset, duration)

        yield chunk_path, offset

        chunk_path.unlink(missing_ok=True)
        offset += duration
        chunk_index += 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_ffmpeg(cmd: list[str], source: Path) -> None:
    """Execute an ffmpeg command, raising RuntimeError on failure.

    Args:
        cmd: Full ffmpeg command as a list of strings.
        source: Source file (used only for error messages).

    Raises:
        RuntimeError: When ffmpeg exits with a non-zero code.
    """
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg failed for {source.name}:\n{stderr}"
        ) from exc
