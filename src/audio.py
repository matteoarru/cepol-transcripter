"""Audio extraction and chunking using ffmpeg subprocesses.

All heavy lifting is delegated to ffmpeg so we never load entire audio files
into Python memory — critical for 4-hour recordings.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Generator, Iterator, Sequence, Tuple

from .config import (
    FFMPEG_CHANNELS,
    FFMPEG_CODEC,
    FFMPEG_SAMPLE_RATE,
    VIDEO_EXTENSIONS,
)

logger = logging.getLogger(__name__)


# Suffix used for the cached audio sidecar produced from video files.
AUDIO_CACHE_SUFFIX = ".audio.wav"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_FFMPEG_DIR = PROJECT_ROOT / ".tools" / "ffmpeg-static"


@dataclass(frozen=True)
class ChunkSpec:
    """One extraction/transcription chunk within a media file."""

    index: int
    offset: float
    duration: float


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def cached_audio_path(source: Path) -> Path:
    """Return the expected sidecar WAV path for a video file.

    The sidecar lives next to the source:
    ``interview.mp4`` → ``interview.audio.wav``

    Args:
        source: Original video file path.

    Returns:
        Sidecar WAV :class:`~pathlib.Path` (may or may not exist yet).
    """
    return source.parent / (source.stem + AUDIO_CACHE_SUFFIX)


def prepare_audio(source: Path) -> tuple[Path, bool]:
    """Return an audio-only WAV path ready for chunking, caching when needed.

    * **Audio files** are returned unchanged — no conversion required.
    * **Video files** are extracted to a sidecar ``*.audio.wav`` once.
      On subsequent calls the sidecar is reused if it exists and is newer
      than the source, avoiding redundant ffmpeg decoding.

    Args:
        source: Input media file (audio or video).

    Returns:
        Tuple of ``(audio_path, was_cached)`` where ``was_cached`` is
        ``True`` when an existing sidecar was reused.

    Raises:
        RuntimeError: When ffmpeg extraction fails.
    """
    if not is_video(source):
        return source, False

    sidecar = cached_audio_path(source)

    if sidecar.exists() and sidecar.stat().st_mtime >= source.stat().st_mtime:
        logger.info("Audio cache hit: %s", sidecar.name)
        return sidecar, True

    logger.info("Extracting audio: %s → %s", source.name, sidecar.name)
    extract_audio(source, sidecar)
    return sidecar, False


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
        _tool_binary("ffprobe"), "-v", "quiet",
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
        _tool_binary("ffmpeg"), "-y",
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
        _tool_binary("ffmpeg"), "-y",
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


def build_chunk_plan(total_duration: float, chunk_size: int) -> list[ChunkSpec]:
    """Return ordered chunk specifications covering *total_duration*."""
    chunks: list[ChunkSpec] = []
    offset = 0.0
    chunk_index = 0

    while offset < total_duration:
        remaining = total_duration - offset
        duration = min(float(chunk_size), remaining)
        chunks.append(ChunkSpec(index=chunk_index, offset=offset, duration=duration))
        offset += duration
        chunk_index += 1

    return chunks


def audio_chunks(
    source: Path,
    chunk_size: int,
    max_duration: float | None = None,
) -> ContextManager[Iterator[Tuple[Path, float]]]:
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
    return audio_chunks_from_plan(
        source,
        _chunk_plan_for_source(source, chunk_size, max_duration),
    )


@contextmanager
def audio_chunks_from_plan(
    source: Path,
    chunk_plan: Sequence[ChunkSpec],
) -> Generator[Iterator[Tuple[Path, float]], None, None]:
    """Yield sequentially extracted chunks for a precomputed plan."""
    with tempfile.TemporaryDirectory(prefix="cepol_chunk_") as tmpdir:
        yield _chunk_iterator(source, Path(tmpdir), chunk_plan)


def pipelined_audio_chunks(
    source: Path,
    chunk_size: int,
    max_duration: float | None = None,
    max_prefetch: int = 5,
) -> ContextManager[Iterator[Tuple[Path, float]]]:
    """Yield chunks while future chunks are extracted in background threads.

    This keeps ffmpeg busy on upcoming chunks while the caller transcribes the
    current one, reducing idle GPU time on long media files.
    """
    return pipelined_audio_chunks_from_plan(
        source,
        _chunk_plan_for_source(source, chunk_size, max_duration),
        max_prefetch=max_prefetch,
    )


@contextmanager
def pipelined_audio_chunks_from_plan(
    source: Path,
    chunk_plan: Sequence[ChunkSpec],
    max_prefetch: int = 5,
) -> Generator[Iterator[Tuple[Path, float]], None, None]:
    """Yield prefetched chunks for a precomputed plan."""
    with tempfile.TemporaryDirectory(prefix="cepol_chunk_") as tmpdir:
        with ThreadPoolExecutor(
            max_workers=max(1, max_prefetch),
            thread_name_prefix="cepol_extract",
        ) as executor:
            yield _prefetched_chunk_iterator(
                source,
                Path(tmpdir),
                chunk_plan,
                executor,
                max(1, max_prefetch),
            )


def _chunk_iterator(
    source: Path,
    tmpdir: Path,
    chunk_plan: Sequence[ChunkSpec],
) -> Iterator[Tuple[Path, float]]:
    """Internal generator that materialises and yields chunks one at a time.

    Each chunk WAV is extracted by ffmpeg, yielded to the caller, then
    immediately deleted before the next chunk is extracted.  This keeps
    peak disk usage to a single chunk at any given moment.

    Args:
        source:         Original media file.
        tmpdir:         Temporary directory (managed by caller).
        chunk_plan: Ordered extraction plan.

    Yields:
        Tuples of (chunk_wav_path, absolute_start_offset_seconds).
    """
    for spec in chunk_plan:
        chunk_path = _extract_chunk_to_path(source, tmpdir, spec)
        yield chunk_path, spec.offset
        chunk_path.unlink(missing_ok=True)


def _prefetched_chunk_iterator(
    source: Path,
    tmpdir: Path,
    chunk_plan: Sequence[ChunkSpec],
    executor: ThreadPoolExecutor,
    max_prefetch: int,
) -> Iterator[Tuple[Path, float]]:
    """Yield extracted chunks in order while prefetching future chunks."""
    futures_by_index: dict[int, Future[Path]] = {}
    next_to_submit = 0

    def submit_next() -> None:
        nonlocal next_to_submit
        if next_to_submit >= len(chunk_plan):
            return
        spec = chunk_plan[next_to_submit]
        futures_by_index[spec.index] = executor.submit(
            _extract_chunk_to_path,
            source,
            tmpdir,
            spec,
        )
        next_to_submit += 1

    for _ in range(min(max_prefetch, len(chunk_plan))):
        submit_next()

    for spec in chunk_plan:
        chunk_path = futures_by_index.pop(spec.index).result()
        submit_next()
        yield chunk_path, spec.offset
        chunk_path.unlink(missing_ok=True)


def _extract_chunk_to_path(source: Path, tmpdir: Path, spec: ChunkSpec) -> Path:
    """Materialise one chunk WAV on disk and return its path."""
    chunk_path = tmpdir / f"chunk_{spec.index:04d}.wav"
    logger.debug(
        "Extracting chunk %d: %.1fs – %.1fs from %s",
        spec.index,
        spec.offset,
        spec.offset + spec.duration,
        source.name,
    )
    extract_audio_segment(source, chunk_path, spec.offset, spec.duration)
    return chunk_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_plan_for_source(
    source: Path,
    chunk_size: int,
    max_duration: float | None,
) -> list[ChunkSpec]:
    """Probe source duration and return a bounded chunk plan."""
    total_duration = get_duration(source)
    if max_duration is not None:
        total_duration = min(total_duration, max_duration)
    return build_chunk_plan(total_duration, chunk_size)


def _tool_binary(tool_name: str) -> str:
    """Return the binary path for ffmpeg-family tools.

    Resolution order:
    1. matching environment variable (`FFMPEG_BIN` or `FFPROBE_BIN`)
    2. tool found on `PATH`
    3. repo-local static binary under `.tools/ffmpeg-static/`
    4. plain command name as a last resort
    """
    env_name = f"{tool_name.upper()}_BIN"
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value

    system_path = shutil.which(tool_name)
    if system_path:
        return system_path

    local_binary = LOCAL_FFMPEG_DIR / tool_name
    if local_binary.is_file():
        return str(local_binary)

    return tool_name


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
