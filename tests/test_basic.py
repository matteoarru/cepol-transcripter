"""Basic unit tests — no GPU or ffmpeg required.

Run with:
    pytest tests/
"""

import sys
from pathlib import Path

# Ensure the project root is on the path when running tests directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.config import (
    AUDIO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    TranscriptionConfig,
)
from src.transcriber import Segment
from src.writer import _format_srt_timestamp, write_srt, write_txt


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_supported_extensions_union(self):
        assert SUPPORTED_EXTENSIONS == AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

    def test_default_config_values(self):
        cfg = TranscriptionConfig()
        assert cfg.model == "large-v3-turbo"
        assert cfg.language == "en"
        assert cfg.device == "cuda"
        assert cfg.vad_filter is True


# ---------------------------------------------------------------------------
# writer — _format_srt_timestamp
# ---------------------------------------------------------------------------

class TestFormatSrtTimestamp:
    def test_zero(self):
        assert _format_srt_timestamp(0.0) == "00:00:00,000"

    def test_one_second(self):
        assert _format_srt_timestamp(1.0) == "00:00:01,000"

    def test_one_minute(self):
        assert _format_srt_timestamp(60.0) == "00:01:00,000"

    def test_one_hour(self):
        assert _format_srt_timestamp(3600.0) == "01:00:00,000"

    def test_fractional(self):
        assert _format_srt_timestamp(1.5) == "00:00:01,500"

    def test_large_value(self):
        # 3h 59m 59.999s
        result = _format_srt_timestamp(3 * 3600 + 59 * 60 + 59.999)
        assert result.startswith("03:59:59,")

    def test_negative_clamped_to_zero(self):
        assert _format_srt_timestamp(-5.0) == "00:00:00,000"


# ---------------------------------------------------------------------------
# writer — write_txt / write_srt
# ---------------------------------------------------------------------------

class TestWriteTxt:
    def test_creates_file(self, tmp_path):
        segs = [Segment(0.0, 1.0, "Hello world.")]
        out = tmp_path / "test.txt"
        write_txt(segs, out)
        assert out.exists()
        assert "Hello world." in out.read_text()

    def test_empty_segments(self, tmp_path):
        out = tmp_path / "empty.txt"
        write_txt([], out)
        assert out.exists()
        assert out.read_text() == "\n"

    def test_paragraph_break_on_long_pause(self, tmp_path):
        segs = [
            Segment(0.0, 1.0, "First."),
            Segment(10.0, 11.0, "Second."),   # gap > 2s → blank line
        ]
        out = tmp_path / "paused.txt"
        write_txt(segs, out)
        content = out.read_text()
        assert "\n\n" in content


class TestWriteSrt:
    def test_creates_file(self, tmp_path):
        segs = [Segment(0.0, 2.5, "Test subtitle.")]
        out = tmp_path / "test.srt"
        write_srt(segs, out)
        assert out.exists()

    def test_srt_format(self, tmp_path):
        segs = [
            Segment(0.0, 1.5, "First line."),
            Segment(2.0, 3.5, "Second line."),
        ]
        out = tmp_path / "test.srt"
        write_srt(segs, out)
        content = out.read_text()
        assert "1\n" in content
        assert "2\n" in content
        assert "00:00:00,000 --> 00:00:01,500" in content
        assert "00:00:02,000 --> 00:00:03,500" in content
        assert "First line." in content
        assert "Second line." in content

    def test_empty_segments(self, tmp_path):
        out = tmp_path / "empty.srt"
        write_srt([], out)
        assert out.exists()
        assert out.read_text() == "\n"


# ---------------------------------------------------------------------------
# scanner — skip logic (mocked filesystem)
# ---------------------------------------------------------------------------

class TestSkipLogic:
    def test_already_processed_skips(self, tmp_path):
        from src.scanner import _already_processed

        media = tmp_path / "video.mp4"
        media.write_bytes(b"fake")

        txt = tmp_path / "video.txt"
        srt = tmp_path / "video.srt"
        txt.write_text("transcript")
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")

        # Make outputs newer than the source
        import os, time as t
        future = t.time() + 10
        os.utime(txt, (future, future))
        os.utime(srt, (future, future))

        assert _already_processed(media) is True

    def test_missing_txt_not_skipped(self, tmp_path):
        from src.scanner import _already_processed

        media = tmp_path / "audio.wav"
        media.write_bytes(b"fake")
        (tmp_path / "audio.srt").write_text("content")

        assert _already_processed(media) is False

    def test_missing_both_not_skipped(self, tmp_path):
        from src.scanner import _already_processed

        media = tmp_path / "audio.wav"
        media.write_bytes(b"fake")

        assert _already_processed(media) is False


# ---------------------------------------------------------------------------
# scanner — iter_media_files
# ---------------------------------------------------------------------------

class TestIterMediaFiles:
    def test_finds_supported_files(self, tmp_path):
        from src.scanner import iter_media_files

        (tmp_path / "a.mp4").write_bytes(b"v")
        (tmp_path / "b.wav").write_bytes(b"a")
        (tmp_path / "c.txt").write_bytes(b"t")   # unsupported

        found = {p.name for p in iter_media_files(tmp_path)}
        assert found == {"a.mp4", "b.wav"}

    def test_recurses_subdirs(self, tmp_path):
        from src.scanner import iter_media_files

        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "clip.mkv").write_bytes(b"v")
        (tmp_path / "root.mp3").write_bytes(b"a")

        found = {p.name for p in iter_media_files(tmp_path)}
        assert found == {"clip.mkv", "root.mp3"}

    def test_raises_on_non_directory(self, tmp_path):
        from src.scanner import iter_media_files

        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            list(iter_media_files(f))
