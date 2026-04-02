"""Basic unit tests — no GPU or ffmpeg required.

Run with:
    pytest tests/
"""

import importlib
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
from src.output_paths import transcript_output_paths
from src.output_paths import transcript_outputs_exist
from src.transcriber import Segment, TranscriptionResult
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

    def test_env_backed_defaults(self, monkeypatch):
        import src.config as config_module

        monkeypatch.setenv("MEDIA_CHUNK_MINUTES", "20")
        monkeypatch.setenv("PIPELINE_THREADS", "8")
        monkeypatch.setenv("CPU_THREADS", "12")
        monkeypatch.setenv("WHISPER_BEAM_SIZE", "2")
        monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8_float16")
        monkeypatch.setenv("WHISPER_VAD", "false")
        monkeypatch.setenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false")
        reloaded = importlib.reload(config_module)

        try:
            assert reloaded.DEFAULT_CHUNK_SIZE_SECONDS == 1200
            assert reloaded.DEFAULT_PIPELINE_THREADS == 5
            assert reloaded.DEFAULT_CPU_THREADS == 12
            assert reloaded.BEAM_SIZE == 2
            assert reloaded.DEFAULT_COMPUTE_TYPE == "int8_float16"
            assert reloaded.TranscriptionConfig().pipeline_threads == 5
            assert reloaded.TranscriptionConfig().vad_filter is False
            assert reloaded.TranscriptionConfig().condition_on_previous_text is False
        finally:
            monkeypatch.delenv("MEDIA_CHUNK_MINUTES", raising=False)
            monkeypatch.delenv("PIPELINE_THREADS", raising=False)
            monkeypatch.delenv("CPU_THREADS", raising=False)
            monkeypatch.delenv("WHISPER_BEAM_SIZE", raising=False)
            monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
            monkeypatch.delenv("WHISPER_VAD", raising=False)
            monkeypatch.delenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT", raising=False)
            importlib.reload(config_module)


# ---------------------------------------------------------------------------
# output paths
# ---------------------------------------------------------------------------

class TestOutputPaths:
    def test_transcript_output_paths_are_siblings(self, tmp_path):
        media = tmp_path / "case01" / "interview.final.mp4"
        media.parent.mkdir(parents=True)

        txt, srt = transcript_output_paths(media)

        assert txt == media.parent / "interview.final.txt"
        assert srt == media.parent / "interview.final.srt"

    def test_transcript_outputs_exist_requires_both_files(self, tmp_path):
        media = tmp_path / "case01" / "interview.final.mp4"
        media.parent.mkdir(parents=True)

        assert transcript_outputs_exist(media) is False

        txt, srt = transcript_output_paths(media)
        txt.write_text("transcript")
        assert transcript_outputs_exist(media) is False

        srt.write_text("subtitle")
        assert transcript_outputs_exist(media) is True


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

        assert _already_processed(media) is True

    def test_already_processed_ignores_source_mtime(self, tmp_path):
        from src.scanner import _already_processed

        media = tmp_path / "video.mp4"
        media.write_bytes(b"fake")
        txt = tmp_path / "video.txt"
        srt = tmp_path / "video.srt"
        txt.write_text("transcript")
        srt.write_text("subtitle")

        import os
        import time as t

        future = t.time() + 10
        os.utime(media, (future, future))

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

    def test_skips_generated_audio_cache_sidecars(self, tmp_path):
        from src.scanner import iter_media_files

        (tmp_path / "lecture.mp4").write_bytes(b"video")
        (tmp_path / "lecture.audio.wav").write_bytes(b"cache")

        found = {p.name for p in iter_media_files(tmp_path)}
        assert found == {"lecture.mp4"}

    def test_raises_on_non_directory(self, tmp_path):
        from src.scanner import iter_media_files

        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            list(iter_media_files(f))


# ---------------------------------------------------------------------------
# audio — chunk planning / pipelining
# ---------------------------------------------------------------------------

class TestAudioChunkPipeline:
    def test_build_chunk_plan(self):
        from src.audio import build_chunk_plan

        plan = build_chunk_plan(total_duration=125.0, chunk_size=60)

        assert [(item.index, item.offset, item.duration) for item in plan] == [
            (0, 0.0, 60.0),
            (1, 60.0, 60.0),
            (2, 120.0, 5.0),
        ]

    def test_pipelined_audio_chunks_yield_in_order(self, monkeypatch, tmp_path):
        from src import audio

        source = tmp_path / "video.mp4"
        source.write_bytes(b"video")

        monkeypatch.setattr(audio, "get_duration", lambda _: 125.0)

        extracted: list[tuple[float, float]] = []

        def fake_extract(source_path, target_path, start_seconds, duration_seconds):
            target_path.write_text(f"{start_seconds}:{duration_seconds}")
            extracted.append((start_seconds, duration_seconds))

        monkeypatch.setattr(audio, "extract_audio_segment", fake_extract)

        with audio.pipelined_audio_chunks(source, chunk_size=60, max_prefetch=3) as chunk_iter:
            observed = [
                (chunk_path.read_text(), offset)
                for chunk_path, offset in chunk_iter
            ]

        assert observed == [
            ("0.0:60.0", 0.0),
            ("60.0:60.0", 60.0),
            ("120.0:5.0", 120.0),
        ]
        assert sorted(extracted) == [
            (0.0, 60.0),
            (60.0, 60.0),
            (120.0, 5.0),
        ]


# ---------------------------------------------------------------------------
# transcriber — runtime helpers
# ---------------------------------------------------------------------------

class TestTranscriberRuntime:
    def test_missing_cuda_runtime_libraries(self, monkeypatch):
        from src import transcriber

        available = {"libcuda.so.1"}
        monkeypatch.setattr(
            transcriber,
            "_can_load_shared_library",
            lambda name: name in available,
        )

        missing = transcriber.missing_cuda_runtime_libraries()

        assert missing == ("libcublas.so.12", "libcudnn.so.9")


# ---------------------------------------------------------------------------
# processor — single-file workflow orchestration
# ---------------------------------------------------------------------------

class TestProcessor:
    def test_process_file_success(self, monkeypatch, tmp_path):
        from src import processor

        class FakeBar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        media = tmp_path / "briefing.wav"
        plan = processor.MediaProcessingPlan(
            duration=120.0,
            processed_duration=90.0,
            chunk_plan=[],
            use_pipeline=True,
            source_strategy="direct_media",
        )
        transcription = TranscriptionResult(
            segments=[Segment(0.0, 1.0, "Briefing started.")],
            duration=90.0,
            language="en",
            elapsed=3.0,
        )
        txt_path = tmp_path / "briefing.txt"
        srt_path = tmp_path / "briefing.srt"

        monkeypatch.setattr(processor, "_build_processing_plan", lambda path, cfg: plan)
        monkeypatch.setattr(processor, "_chunk_progress_bar", lambda filename, duration: FakeBar())
        monkeypatch.setattr(processor, "load_model", lambda cfg: object())
        monkeypatch.setattr(
            processor,
            "_transcribe_media",
            lambda media_path, model, plan_obj, cfg, chunk_bar, metrics: transcription,
        )
        monkeypatch.setattr(
            processor,
            "_write_outputs",
            lambda media_path, transcript: (txt_path, srt_path),
        )
        result = processor.process_file(media, TranscriptionConfig())

        assert result.ok is True
        assert result.media_path == media
        assert result.error is None
        assert result.txt_path == txt_path
        assert result.srt_path == srt_path
        assert result.source_strategy == "direct_media"
        assert result.metrics is not None

    def test_process_file_failure(self, monkeypatch, tmp_path):
        from src import processor

        class FakeBar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        media = tmp_path / "briefing.wav"
        plan = processor.MediaProcessingPlan(
            duration=120.0,
            processed_duration=90.0,
            chunk_plan=[],
            use_pipeline=False,
            source_strategy="prepared_audio",
        )

        monkeypatch.setattr(processor, "_build_processing_plan", lambda path, cfg: plan)
        monkeypatch.setattr(processor, "_chunk_progress_bar", lambda filename, duration: FakeBar())
        monkeypatch.setattr(processor, "load_model", lambda cfg: object())

        def boom(*args, **kwargs):
            raise RuntimeError("transcription failed")

        monkeypatch.setattr(processor, "_transcribe_media", boom)

        result = processor.process_file(media, TranscriptionConfig())

        assert result.ok is False
        assert result.media_path == media
        assert result.error == "transcription failed"
        assert result.txt_path is None
        assert result.srt_path is None

    def test_process_file_retries_on_cpu_after_cuda_failure(self, monkeypatch, tmp_path):
        from src import processor

        media = tmp_path / "briefing.wav"
        txt_path = tmp_path / "briefing.txt"
        srt_path = tmp_path / "briefing.srt"
        plan = processor.MediaProcessingPlan(
            duration=30.0,
            processed_duration=30.0,
            chunk_plan=[],
            use_pipeline=False,
            source_strategy="direct_media",
        )
        transcription = TranscriptionResult(
            segments=[Segment(0.0, 1.0, "Recovered on CPU.")],
            duration=30.0,
            language="en",
            elapsed=2.0,
        )
        calls: list[tuple[str, str]] = []

        def fake_load_model(cfg):
            return f"model:{cfg.device}"

        def fake_process_once(media_path, cfg, model, metrics=None):
            calls.append((cfg.device, cfg.compute_type))
            if cfg.device == "cuda":
                raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
            return transcription, txt_path, srt_path, plan

        monkeypatch.setattr(processor, "load_model", fake_load_model)
        monkeypatch.setattr(processor, "_process_file_once", fake_process_once)

        result = processor.process_file(media, TranscriptionConfig())

        assert result.ok is True
        assert result.txt_path == txt_path
        assert result.srt_path == srt_path
        assert calls == [("cuda", "float16"), ("cpu", "int8")]

    def test_process_file_removes_audio_cache_after_success(self, monkeypatch, tmp_path):
        from src import processor

        class FakeBar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        media = tmp_path / "briefing.mp4"
        media.write_bytes(b"video")
        sidecar = tmp_path / "briefing.audio.wav"
        sidecar.write_bytes(b"cache")
        txt_path = tmp_path / "briefing.txt"
        srt_path = tmp_path / "briefing.srt"
        plan = processor.MediaProcessingPlan(
            duration=30.0,
            processed_duration=30.0,
            chunk_plan=[],
            use_pipeline=False,
            source_strategy="prepared_audio",
        )
        transcription = TranscriptionResult(
            segments=[Segment(0.0, 1.0, "Completed.")],
            duration=30.0,
            language="en",
            elapsed=2.0,
        )

        monkeypatch.setattr(processor, "_build_processing_plan", lambda path, cfg: plan)
        monkeypatch.setattr(processor, "_chunk_progress_bar", lambda filename, duration: FakeBar())
        monkeypatch.setattr(processor, "load_model", lambda cfg: object())
        monkeypatch.setattr(
            processor,
            "_transcribe_media",
            lambda media_path, model, plan_obj, cfg, chunk_bar, metrics: transcription,
        )
        monkeypatch.setattr(
            processor,
            "_write_outputs",
            lambda media_path, transcript: (txt_path, srt_path),
        )

        result = processor.process_file(media, TranscriptionConfig())

        assert result.ok is True
        assert sidecar.exists() is False

    def test_build_processing_plan_prefers_cached_audio_for_long_video(self, monkeypatch, tmp_path):
        from src import processor

        media = tmp_path / "briefing.mp4"
        media.write_bytes(b"video")
        cache = tmp_path / "briefing.audio.wav"
        cache.write_bytes(b"cache")

        monkeypatch.setattr(processor, "get_duration", lambda path: 180.0)

        plan = processor._build_processing_plan(
            media,
            TranscriptionConfig(chunk_size=60, pipeline_threads=3),
        )

        assert plan.use_pipeline is True
        assert plan.source_strategy == "cached_audio"

    def test_worker_slot_uses_process_identity(self, monkeypatch):
        from src import processor

        class FakeProcess:
            _identity = (3,)
            name = "ForkProcess-3"

        monkeypatch.setattr(processor, "current_process", lambda: FakeProcess())

        assert processor._worker_slot(2) == 0
