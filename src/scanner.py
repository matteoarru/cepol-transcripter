"""Recursive file-system scanner with skip-logic for already-processed files."""

import logging
from pathlib import Path
from typing import Iterator

from .audio import AUDIO_CACHE_SUFFIX
from .config import SUPPORTED_EXTENSIONS
from .output_paths import transcript_output_paths

logger = logging.getLogger(__name__)


def iter_media_files(root: Path) -> Iterator[Path]:
    """Yield every supported media file found recursively under *root*.

    Files whose sibling .txt AND .srt outputs are both newer than the source
    are skipped (resume / skip logic).

    Args:
        root: Directory to search recursively.

    Yields:
        Absolute :class:`~pathlib.Path` objects for each pending media file.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if not path.is_file():
            continue
        if _is_generated_audio_cache(path):
            logger.debug("SKIP (generated audio cache): %s", path.name)
            continue
        if _already_processed(path):
            logger.info("SKIP (already processed): %s", path.name)
            continue
        yield path


def _already_processed(media_path: Path) -> bool:
    """Return True when both output files exist and are newer than the source.

    Args:
        media_path: Path to the media file being evaluated.

    Returns:
        True if processing can safely be skipped.
    """
    txt, srt = transcript_output_paths(media_path)
    if not txt.exists() or not srt.exists():
        return False
    source_mtime = media_path.stat().st_mtime
    return txt.stat().st_mtime > source_mtime and srt.stat().st_mtime > source_mtime


def _is_generated_audio_cache(path: Path) -> bool:
    """Return True for generated ``*.audio.wav`` cache sidecars."""
    return path.name.endswith(AUDIO_CACHE_SUFFIX)


def count_media_files(root: Path) -> int:
    """Return the total number of pending media files under *root*.

    Args:
        root: Directory to scan.

    Returns:
        Integer count of pending files.
    """
    return sum(1 for _ in iter_media_files(root))
