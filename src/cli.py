"""Command-line interface — argument parsing only, no business logic."""

import argparse
from pathlib import Path

from .config import (
    DEFAULT_CHUNK_SIZE_SECONDS,
    DEFAULT_MODEL,
    DEFAULT_VAD,
    DEFAULT_WORKERS,
    TranscriptionConfig,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the configured :class:`~argparse.ArgumentParser`.

    Returns:
        Argument parser for the transcription CLI.
    """
    parser = argparse.ArgumentParser(
        prog="cepol-transcripter",
        description=(
            "Recursively transcribe audio/video files using faster-whisper "
            "(CUDA). Optimised for law enforcement English recordings."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root folder to scan recursively for media files.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="NAME",
        help=(
            "faster-whisper model name. "
            "Options: tiny, base, small, medium, large-v2, large-v3, "
            "large-v3-turbo (default, fastest)."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=(
            "Number of parallel file-processing workers. "
            "Values > 1 load the model in each worker process; only "
            "recommended when multiple GPUs are available."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE_SECONDS,
        metavar="SECONDS",
        dest="chunk_size",
        help="Maximum audio chunk length in seconds for long files.",
    )

    parser.add_argument(
        "--no-vad",
        action="store_false",
        dest="vad",
        default=DEFAULT_VAD,
        help="Disable VAD (Voice Activity Detection) filtering.",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        dest="max_duration",
        help="Hard cap on total processed duration per file (for testing).",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )

    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "int8_float16", "int8", "float32"],
        dest="compute_type",
        help="Quantisation / compute type for the model.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity.",
    )

    return parser


def parse_args() -> tuple[argparse.Namespace, TranscriptionConfig]:
    """Parse CLI arguments and return (namespace, TranscriptionConfig).

    Returns:
        Tuple of the raw :class:`~argparse.Namespace` and a fully populated
        :class:`~config.TranscriptionConfig`.
    """
    parser = build_parser()
    args = parser.parse_args()

    config = TranscriptionConfig(
        model=args.model,
        compute_type=args.compute_type,
        device=args.device,
        workers=args.workers,
        chunk_size=args.chunk_size,
        vad_filter=args.vad,
        max_duration=args.max_duration,
    )

    return args, config
