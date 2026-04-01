"""Central configuration — all constants and runtime settings live here."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# .env loader — sets os.environ for any key not already present in the shell
# ---------------------------------------------------------------------------

def load_env(env_path: Path | None = None) -> None:
    """Load variables from a .env file into os.environ without overwriting.

    Shell-set variables always take precedence over .env values.  If the .env
    file is absent the function is a no-op, so the app works in environments
    (CI, Docker) that inject secrets via the shell.

    Args:
        env_path: Path to the .env file.  Defaults to ``<project_root>/.env``.
    """
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"

    if not env_path.is_file():
        return

    with env_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Only set if the variable is not already in the environment
            if key and key not in os.environ and value:
                os.environ[key] = value


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read an integer from the environment, clamping when bounds are set."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_csv(name: str) -> tuple[str, ...]:
    """Read a comma-separated list from the environment."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    """Read a string choice from the environment, falling back when invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw if raw in allowed else default


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
ALLOWED_COMPUTE_TYPES: set[str] = {"float16", "int8_float16", "int8", "float32"}
DEFAULT_COMPUTE_TYPE: str = _env_choice(
    "WHISPER_COMPUTE_TYPE",
    "float16",
    ALLOWED_COMPUTE_TYPES,
)
DEFAULT_DEVICE: str = "cuda"
DEFAULT_LANGUAGE: str = "en"
DEFAULT_CHUNK_SIZE_SECONDS: int = (
    _env_int("MEDIA_CHUNK_MINUTES", 20, minimum=1) * 60
)
DEFAULT_WORKERS: int = 1
DEFAULT_VAD: bool = _env_bool("WHISPER_VAD", True)
DEFAULT_CPU_THREADS: int = _env_int(
    "CPU_THREADS",
    min(8, max(1, os.cpu_count() or 4)),
    minimum=1,
)
MAX_PIPELINE_THREADS: int = 5
DEFAULT_PIPELINE_THREADS: int = _env_int(
    "PIPELINE_THREADS",
    MAX_PIPELINE_THREADS,
    minimum=1,
    maximum=MAX_PIPELINE_THREADS,
)
VISIBLE_GPU_IDS: tuple[str, ...] = _env_csv("CUDA_VISIBLE_DEVICES")


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

BEAM_SIZE: int = _env_int("WHISPER_BEAM_SIZE", 5, minimum=1)
BEST_OF: int = _env_int("WHISPER_BEST_OF", 1, minimum=1)
TEMPERATURE: float = 0.0
NO_SPEECH_THRESHOLD: float = 0.6
COMPRESSION_RATIO_THRESHOLD: float = 2.4
LOG_PROB_THRESHOLD: float = -1.0
CONDITION_ON_PREVIOUS_TEXT: bool = _env_bool(
    "WHISPER_CONDITION_ON_PREVIOUS_TEXT",
    True,
)


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
    device_index: int = 0
    language: str = DEFAULT_LANGUAGE
    visible_gpu_ids: tuple[str, ...] = VISIBLE_GPU_IDS
    chunk_size: int = DEFAULT_CHUNK_SIZE_SECONDS
    workers: int = DEFAULT_WORKERS
    cpu_threads: int = DEFAULT_CPU_THREADS
    pipeline_threads: int = DEFAULT_PIPELINE_THREADS
    vad_filter: bool = DEFAULT_VAD
    max_duration: Optional[float] = None
    beam_size: int = BEAM_SIZE
    best_of: int = BEST_OF
    temperature: float = TEMPERATURE
    condition_on_previous_text: bool = CONDITION_ON_PREVIOUS_TEXT
