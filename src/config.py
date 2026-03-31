"""Central configuration — all constants and runtime settings live here."""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".m4a", ".flac"})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mkv", ".avi", ".mov"})
SUPPORTED_EXTENSIONS: frozenset[str] = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Default transcription settings
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "large-v3-turbo"
DEFAULT_COMPUTE_TYPE: str = "float16"
DEFAULT_DEVICE: str = "cuda"
DEFAULT_LANGUAGE: str = "en"
DEFAULT_CHUNK_SIZE_SECONDS: int = 1800   # 30 min per chunk
DEFAULT_WORKERS: int = 1
DEFAULT_VAD: bool = True


# ---------------------------------------------------------------------------
# ffmpeg audio normalisation target
# ---------------------------------------------------------------------------

FFMPEG_SAMPLE_RATE: int = 16_000
FFMPEG_CHANNELS: int = 1
FFMPEG_CODEC: str = "pcm_s16le"


# ---------------------------------------------------------------------------
# VAD parameters — tuned for law-enforcement recordings
# ---------------------------------------------------------------------------

VAD_PARAMETERS: dict = {
    "threshold": 0.45,
    "min_speech_duration_ms": 200,
    "min_silence_duration_ms": 600,
    "speech_pad_ms": 400,
    "max_speech_duration_s": float("inf"),
}


# ---------------------------------------------------------------------------
# Whisper inference settings — optimised for speed on single GPU
# ---------------------------------------------------------------------------

BEAM_SIZE: int = 5
BEST_OF: int = 1
TEMPERATURE: float = 0.0
NO_SPEECH_THRESHOLD: float = 0.6
COMPRESSION_RATIO_THRESHOLD: float = 2.4
LOG_PROB_THRESHOLD: float = -1.0


# ---------------------------------------------------------------------------
# Runtime configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionConfig:
    """All runtime knobs for the transcription pipeline.

    Instantiated once from CLI arguments and passed through the call stack.
    """

    model: str = DEFAULT_MODEL
    compute_type: str = DEFAULT_COMPUTE_TYPE
    device: str = DEFAULT_DEVICE
    language: str = DEFAULT_LANGUAGE
    chunk_size: int = DEFAULT_CHUNK_SIZE_SECONDS
    workers: int = DEFAULT_WORKERS
    vad_filter: bool = DEFAULT_VAD
    max_duration: Optional[float] = None
    beam_size: int = BEAM_SIZE
    best_of: int = BEST_OF
    temperature: float = TEMPERATURE
