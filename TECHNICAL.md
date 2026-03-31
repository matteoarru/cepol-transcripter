# CEPOL Transcripter — Technical Notes

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
- load one model per process
- dispatch per-file work
- print the final summary

### `src/config.py`

Owns runtime configuration and environment-backed defaults:

- `.env` loading
- integer env parsing and clamping
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
- drive chunk-level progress reporting
- call `transcribe_file`
- write output files
- return a structured processing result

### `src/audio.py`

Owns ffprobe/ffmpeg integration:

- probe duration
- extract cached sidecar audio for videos when beneficial
- build chunk plans
- stream sequential chunks
- prefetch future chunks in background threads for long media

### `src/transcriber.py`

Owns faster-whisper integration:

- load the model
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
- "already processed" checks

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
4. It chooses one of two chunk strategies:
   - Sequential path: prepare cached sidecar audio for videos, then extract
     chunks one at a time.
   - Pipelined path: keep the original media as the source and prefetch future
     chunk WAV files in background threads while the current chunk is being
     transcribed.
5. `transcriber.py` receives `(chunk_path, offset)` pairs and returns absolute
   transcript segments.
6. `writer.py` writes sibling `.txt` and `.srt` files.
7. `main.py` includes the saved output paths in the final summary.

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

For videos, the app can reuse a cached sibling `*.audio.wav` sidecar when it is
already newer than the source video. This avoids repeated full-audio extraction
across runs.

### Pipelined

Used for long media when more than one pipeline thread is configured.

The app submits future chunk extractions to a bounded `ThreadPoolExecutor`.
Chunks are still yielded to the transcriber in source order, but ffmpeg can
work ahead while Whisper is busy on the current chunk. This reduces GPU idle
time without requiring multiple model instances.

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

### Small, Named Functions

The per-file workflow is expressed through focused helpers such as:

- `_build_processing_plan`
- `_transcribe_media`
- `_prepare_chunk_source`
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
