"""Helpers for transcript output file locations."""

from pathlib import Path


def transcript_output_paths(media_path: Path) -> tuple[Path, Path]:
    """Return sibling transcript paths for a media file.

    The outputs always live next to the source file and keep the same basename,
    changing only the extension:

    ``/recordings/case01/interview.mp4`` ->
    ``/recordings/case01/interview.txt``
    ``/recordings/case01/interview.srt``
    """
    return media_path.with_suffix(".txt"), media_path.with_suffix(".srt")
