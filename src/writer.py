"""Output writers — produces .txt plain-text and .srt subtitle files."""

import logging
from pathlib import Path
from typing import List

from .transcriber import Segment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_txt(segments: List[Segment], output_path: Path) -> None:
    """Write a plain-text transcript to *output_path*.

    Each segment's text is separated by a single newline.  An empty line is
    inserted between segments whose gap exceeds 2 seconds to preserve the
    natural paragraph structure of speech.

    Args:
        segments:    Ordered list of transcription segments.
        output_path: Destination .txt file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    prev_end: float = 0.0
    for seg in segments:
        # Insert a blank line for pauses longer than 2 seconds
        if lines and (seg.start - prev_end) > 2.0:
            lines.append("")
        lines.append(seg.text)
        prev_end = seg.end

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.debug("TXT written: %s (%d lines)", output_path.name, len(lines))


def write_srt(segments: List[Segment], output_path: Path) -> None:
    """Write an SRT subtitle file to *output_path*.

    Follows the standard SRT format::

        1
        00:00:01,000 --> 00:00:04,500
        First subtitle text.

        2
        00:00:05,000 --> 00:00:08,200
        Second subtitle text.

    Args:
        segments:    Ordered list of transcription segments.
        output_path: Destination .srt file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks: List[str] = []

    for index, seg in enumerate(segments, start=1):
        start_ts = _format_srt_timestamp(seg.start)
        end_ts = _format_srt_timestamp(seg.end)
        blocks.append(f"{index}\n{start_ts} --> {end_ts}\n{seg.text}")

    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    logger.debug("SRT written: %s (%d entries)", output_path.name, len(blocks))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_srt_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to SRT timestamp format.

    SRT timestamps use a comma as the millisecond separator:
    ``HH:MM:SS,mmm``

    Args:
        seconds: Time in seconds (may be fractional).

    Returns:
        Formatted timestamp string, e.g. ``01:23:04,567``.
    """
    seconds = max(0.0, seconds)
    millis = round(seconds * 1000)
    hh, remainder = divmod(millis, 3_600_000)
    mm, remainder = divmod(remainder, 60_000)
    ss, ms = divmod(remainder, 1_000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
