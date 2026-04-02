# CEPOL Transcripter — Technical Notes

Release covered by this document: `v2.0.0`

## Overview

The application is a batch transcription pipeline for audio and video evidence.
It scans a root folder, skips already-processed media, streams media through
ffmpeg in bounded chunks, transcribes those chunks with faster-whisper, and
writes sibling `.txt` and `.srt` files next to the source media.

The current design emphasizes:

- Clear module boundaries.
- Small refactors backed by tests.
- Explicit configuration instead of hidden runtime state.
- Bounded memory and disk usage for long recordings.
- Fast startup failure detection when the CUDA runtime is broken.
- Reusing loaded models instead of reloading them for every file.

## Architecture

```text
main.py
  -> config.load_env()
  -> cli.parse_args()
  -> scanner.iter_media_files()
  -> processor.process_file()
       -> audio / transcriber / writer / output_paths
```

Per-file processing is intentionally isolated in `src/processor.py`, so the
entry point stays focused on orchestration and summary reporting.

## Module Responsibilities

### `main.py`

Owns only top-level orchestration:

- configure logging
- scan for pending files
- probe the runtime once at startup and normalize unsafe device settings
- load one model per process or per worker
- dispatch per-file work with worker-local model caching
- print the final summary

### `src/config.py`

Owns runtime configuration and environment-backed defaults:

- `.env` loading
- integer / boolean / CSV env parsing and clamping
- transcription defaults
- ffmpeg and inference constants

### `src/cli.py`

Owns command-line parsing only:

- argument definitions
- CLI validation
- construction of `TranscriptionConfig`

### `src/processor.py`

Owns the single-file workflow:

- inspect media duration
- choose sequential vs pipelined chunk extraction
- choose whether chunks should come from direct media, prepared audio, or cached sidecar audio
- drive chunk-level progress reporting
- call `transcribe_file`
- capture per-file performance timings
- write output files
- return a structured processing result

### `src/audio.py`

Owns ffprobe/ffmpeg integration:

- probe duration
- extract cached sidecar audio for videos when beneficial
- build chunk plans
- stream sequential chunks
- prefetch future chunks in background threads for long media
- report extraction timings back to the processor

### `src/transcriber.py`

Owns faster-whisper integration:

- load the model
- bootstrap Python-packaged CUDA runtime libraries when present
- detect missing CUDA runtime libraries
- warm up the model with a tiny startup probe when CUDA is requested
- transcribe a chunk
- normalize chunk-relative timestamps into source-relative timestamps
- aggregate chunk results into `TranscriptionResult`

### `src/writer.py`

Owns output formatting only:

- plain-text transcript generation
- SRT timestamp formatting
- subtitle file generation

### `src/scanner.py`

Owns filesystem scanning and skip logic only:

- recursive media discovery
- extension filtering
- "already processed" checks based on sibling `.txt` and `.srt` existence

### `src/output_paths.py`

Owns the output filename rule:

- source and transcript outputs are siblings
- basename is preserved
- only the extension changes

## Processing Flow

For each pending media file:

1. `processor.py` probes the media duration once.
2. It computes the effective processing duration, applying `--max-duration`
   when present.
3. It builds a chunk plan from that duration and the configured chunk size.
4. It chooses a chunk source and execution strategy:
   - `prepared_audio`: for shorter videos, extract or reuse one sibling
     `*.audio.wav` sidecar first, then chunk that normalized audio source.
   - `cached_audio`: for long videos with an already-valid sidecar cache,
     keep the pipeline but extract chunks from the cached WAV instead of
     repeatedly seeking inside the video container.
   - `direct_media`: for long first-run media, chunk directly from the source
     so ffmpeg extraction can overlap with Whisper inference immediately.
5. `transcriber.py` receives `(chunk_path, offset)` pairs and returns absolute
   transcript segments.
6. `writer.py` writes sibling `.txt` and `.srt` files.
7. `processor.py` records per-file timings for probing, preparation,
   extraction, transcription, and writing.
8. `main.py` includes the saved output paths and aggregate timing totals in the
   final summary.

## Chunking Strategy

The pipeline uses ffmpeg temp WAV files rather than loading entire recordings
into Python memory.

Benefits:

- Peak memory remains proportional to chunk size, not total file length.
- ffmpeg handles all decoding and resampling.
- The transcription layer only deals with normalized chunk WAV files.

Chunk defaults are environment-backed:

- `MEDIA_CHUNK_MINUTES=20`
- `PIPELINE_THREADS=5`

The CLI can still override these defaults with `--chunk-size` and
`--pipeline-threads`.

## Sequential vs Pipelined Extraction

### Sequential

Used for shorter files or when chunk prefetching is disabled.

For videos, the app can reuse a sibling `*.audio.wav` sidecar within the current
processing flow when one is already present, for example after an interrupted
run. Successful runs clean up that sidecar after writing `.txt` and `.srt`
outputs.

### Pipelined

Used for long media when more than one pipeline thread is configured.

The app submits future chunk extractions to a bounded `ThreadPoolExecutor`.
Chunks are still yielded to the transcriber in source order, but ffmpeg can
work ahead while Whisper is busy on the current chunk. This reduces GPU idle
time without requiring multiple model instances.

When a sibling `*.audio.wav` sidecar already exists for a long video, the
pipeline adapts and uses that sidecar as the chunk source. That preserves the
overlap benefits of pipelining while avoiding repeated video decode work during
the active run.

## Runtime Preparation

Before the first real media file starts, `main.py` runs a one-time runtime
preparation step.

If `device=cuda`:

- it checks which CUDA devices are visible
- it checks whether the required userspace libraries can be loaded
- it warms up the model with a tiny silent transcription

If any of those checks fail with a CUDA runtime error, the app falls back once
to `device=cpu` and `compute_type=int8` instead of failing repeatedly per file.

## Multi-worker Model Reuse

The original multi-process path submitted one file per task and each task could
load its own model. The current design uses worker initialization:

- one worker process starts
- that worker loads one model once
- all files executed by that worker reuse the same model

In CUDA mode, workers are pinned to visible GPUs through `device_index`, and
the worker count is capped to the number of visible GPUs. In CPU mode, the app
automatically reduces `cpu_threads` per worker to avoid oversubscribing the
host.

## Performance Metrics

Each successful file logs a timing breakdown:

- model load
- ffprobe / planning
- source preparation
- chunk extraction
- transcription
- output writing

The final summary aggregates those timings across the whole batch. Extraction
and transcription totals are intentionally reported separately even though they
can overlap during pipelined runs.

## Benchmark Notes

The `v2.0.0` release adds benchmark-backed tuning guidance for batch servers.

Local benchmark environment:

- Ubuntu 24.04.4 LTS
- NVIDIA RTX 5000 Ada Generation with 32 GB VRAM
- NVIDIA driver `570.211.01`, CUDA `12.8`
- `ffmpeg 6.1.1`
- model: `large-v3-turbo`

Method:

- process two isolated copies of the 74.9-minute sample media in one run
- keep `workers=1` and `pipeline_threads=5`
- warm model downloads ahead of time
- compare runtime and decoding settings, not model size

Findings:

- `--no-condition-on-previous-text` produced the largest single speedup.
- `int8_float16` beat `float16` once decoding was simplified.
- `--no-vad` was fastest on the benchmark sample because the content was dense speech.
- ffmpeg extraction remained a much smaller cost than transcription.

Measured top results:

| Configuration | Wall time | Throughput |
|---------------|-----------|------------|
| `int8_float16 + beam_size=1 + no_condition_on_previous_text + no_vad` | `54.4s` | `165.2x` |
| `float16 + beam_size=1 + no_condition_on_previous_text + no_vad` | `59.2s` | `151.8x` |
| `int8_float16 + beam_size=1 + no_condition_on_previous_text` | `65.1s` | `138.1x` |
| previous baseline defaults | `90.0s` | `99.9x` |

That is why the documentation now recommends two explicit profiles:

- fastest profile for dense-speech batch jobs
- safer throughput profile for mixed real-world corpora

## Output Rules

Outputs always live next to the source media file:

```text
/cases/42/interview.mp4
/cases/42/interview.txt
/cases/42/interview.srt
```

This rule is centralized in `src/output_paths.py`, and both writing and skip
logic use the same helper. That avoids duplication and keeps naming behavior
consistent.

Generated `*.audio.wav` files are treated as transient working files. After a
successful transcription, `src/processor.py` removes the sidecar so only the
source media plus final `.txt` and `.srt` outputs remain.

## Error Handling

The code uses a layered approach:

- `audio.py` raises `RuntimeError` with ffmpeg/ffprobe context.
- `transcriber.py` lets inference errors propagate.
- `processor.py` catches per-file failures and converts them into a structured
  `FileProcessingResult`.
- `main.py` keeps processing the remaining files and prints a final summary.

One bad media file should not stop the rest of the batch.

## Clean Code and XP Practices

The recent refactor intentionally applied a few simple rules.

### Single Responsibility

The main orchestration code and the per-file workflow now live in different
modules (`main.py` and `src/processor.py`). That keeps the entry point small
and makes the single-file workflow easier to test in isolation.

### DRY

Shared rules are centralized:

- output naming lives in `src/output_paths.py`
- environment integer parsing lives in `src/config.py`
- chunk-plan construction lives in `src/audio.py`
- runtime probing lives in `src/transcriber.py`

### Small, Named Functions

The per-file workflow is expressed through focused helpers such as:

- `_build_processing_plan`
- `_transcribe_media`
- `_resolve_chunk_source`
- `_write_outputs`
- `_log_processing_complete`

These names make the control flow readable without needing large comments.

### Refactor with Safety Nets

The codebase keeps a lightweight test suite for pure logic and orchestration
edges. The working rule is:

1. add or keep a small test around a behavior
2. refactor to improve structure
3. rerun `pytest` and `pyright`

That is the XP loop used here for safe incremental change.

## Verification

Recommended local checks:

```bash
venv/bin/python -m pytest -q
venv/bin/python -m pyright
```

These checks are fast enough to run after each refactor, which keeps the cost
of cleanups low.
