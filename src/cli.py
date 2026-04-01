"""Command-line interface — argument parsing only, no business logic."""

import argparse
from pathlib import Path

from . import __version__
from .config import (
    DEFAULT_CHUNK_SIZE_SECONDS,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_CPU_THREADS,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_PIPELINE_THREADS,
    DEFAULT_VAD,
    DEFAULT_WORKERS,
    ALLOWED_COMPUTE_TYPES,
    BEAM_SIZE,
    BEST_OF,
    CONDITION_ON_PREVIOUS_TEXT,
    MAX_PIPELINE_THREADS,
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
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
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
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        metavar="N",
        dest="cpu_threads",
        help=(
            "CPU threads used by CTranslate2. "
            "In CPU mode this is the main throughput knob; in multi-worker CPU "
            "runs it is divided across workers."
        ),
    )

    parser.add_argument(
        "--pipeline-threads",
        type=int,
        default=DEFAULT_PIPELINE_THREADS,
        metavar="N",
        dest="pipeline_threads",
        help=(
            "Background extraction threads for long media pipelines. "
            f"Clamped to 1-{MAX_PIPELINE_THREADS}."
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
        "--beam-size",
        type=int,
        default=BEAM_SIZE,
        metavar="N",
        dest="beam_size",
        help="Beam size for decoding. Lower values are faster but may reduce quality.",
    )

    parser.add_argument(
        "--best-of",
        type=int,
        default=BEST_OF,
        metavar="N",
        dest="best_of",
        help="Number of candidates when sampling-based decoding is used.",
    )

    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_false",
        dest="condition_on_previous_text",
        default=CONDITION_ON_PREVIOUS_TEXT,
        help=(
            "Disable cross-chunk text conditioning. This can improve speed and "
            "reduce error propagation between chunks."
        ),
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
        default=DEFAULT_DEVICE,
        choices=["cuda", "cpu"],
        help="Inference device.",
    )

    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        choices=sorted(ALLOWED_COMPUTE_TYPES),
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
        cpu_threads=max(1, args.cpu_threads),
        pipeline_threads=max(1, min(args.pipeline_threads, MAX_PIPELINE_THREADS)),
        chunk_size=args.chunk_size,
        vad_filter=args.vad,
        max_duration=args.max_duration,
        beam_size=max(1, args.beam_size),
        best_of=max(1, args.best_of),
        condition_on_previous_text=args.condition_on_previous_text,
    )

    return args, config
