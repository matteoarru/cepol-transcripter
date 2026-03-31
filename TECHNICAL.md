# CEPOL Transcripter — Technical & Architecture Documentation

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Responsibilities](#module-responsibilities)
3. [Processing Pipeline](#processing-pipeline)
4. [Chunking Strategy for Long Media](#chunking-strategy-for-long-media)
5. [GPU Usage and Optimisation](#gpu-usage-and-optimisation)
6. [Error Handling Strategy](#error-handling-strategy)
7. [Skip / Resume Logic](#skip--resume-logic)
8. [Data Flow](#data-flow)
9. [Key Design Decisions and Trade-offs](#key-design-decisions-and-trade-offs)
10. [XP and Clean Code Principles Applied](#xp-and-clean-code-principles-applied)
11. [Scalability Considerations](#scalability-considerations)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         CLI (main.py + src/cli.py)                       │
│  argparse → TranscriptionConfig → run()                                  │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │       scanner.py            │
              │  iter_media_files(root)      │
              │  skip-logic (_already_processed) │
              └──────────────┬──────────────┘
                             │  List[Path]
              ┌──────────────▼──────────────┐
              │       audio.py              │
              │  get_duration()  (ffprobe)  │
              │  audio_chunks()  (ffmpeg)   │
              │  extract_audio_segment()    │
              └──────────────┬──────────────┘
                             │  (chunk_path, time_offset) per chunk
              ┌──────────────▼──────────────┐
              │     transcriber.py          │
              │  load_model()               │
              │  transcribe_chunk()         │──→  faster-whisper (CUDA)
              │  transcribe_file()          │
              └──────────────┬──────────────┘
                             │  TranscriptionResult
              ┌──────────────▼──────────────┐
              │       writer.py             │
              │  write_txt()   → .txt       │
              │  write_srt()   → .srt       │
              └─────────────────────────────┘
```

---

## Module Responsibilities

### `src/config.py`
Single source of truth for every constant and runtime parameter.  No business
logic.  All other modules import from here — never define magic numbers
elsewhere.

### `src/scanner.py`
Pure I/O: walks the filesystem, filters by extension, and implements the skip
predicate.  Has no knowledge of transcription or audio processing.

### `src/audio.py`
Wraps ffmpeg via `subprocess`.  Responsible for:
- Probing media duration (ffprobe JSON output)
- Extracting and normalising audio (16 kHz, mono, PCM WAV)
- Splitting media into fixed-length chunks on disk without buffering the full
  file in Python memory

### `src/transcriber.py`
Thin wrapper around `faster_whisper.WhisperModel`.  Responsible for:
- Loading the model once per process
- Transcribing a single chunk wav file
- Applying the `time_offset` to produce absolute timestamps
- Returning strongly-typed `Segment` objects

### `src/writer.py`
Pure output formatting.  Knows nothing about audio or models.  Receives a list
of `Segment` objects and writes standard `.txt` and `.srt` files.

### `src/cli.py`
Argument parsing only.  Returns `(Namespace, TranscriptionConfig)`.  No side
effects beyond `sys.argv` parsing.

### `main.py`
Orchestration only.  Calls the other modules in the correct order, manages the
global progress bar, collects results, and prints the summary.

---

## Processing Pipeline

```
   media file
       │
       ▼
 ffprobe → duration
       │
       ▼
 duration > chunk_size?
       │
  YES  │  NO
  ┌────┘  └─────────────────────┐
  │                              │
  ▼                              ▼
ffmpeg -ss -t (per chunk)    ffmpeg full extract
  → temp .wav                   → temp .wav
       │                              │
       └──────────┬───────────────────┘
                  ▼
        WhisperModel.transcribe()
        (faster-whisper on GPU)
                  │
                  ▼
        segments + time_offset
                  │
                  ▼
        merge all chunk segments
        (ordered by start time)
                  │
          ┌───────┴────────┐
          ▼                ▼
      write_txt()      write_srt()
       → .txt            → .srt
```

---

## Chunking Strategy for Long Media

### Problem

A 4-hour recording at 16 kHz mono 16-bit PCM requires ~460 MB of raw audio
in memory.  Loading this entirely into Python before passing it to the model
would exhaust RAM for concurrent runs.  Additionally, CTranslate2 processes
audio up to ~30 minutes comfortably on 32 GB VRAM; longer inputs may cause
internal attention memory spikes.

### Solution: ffmpeg Fast-Seek Extraction

Each chunk is materialised as a temporary WAV file on disk:

```python
ffmpeg -ss <start> -i <source> -t <chunk_size> -ar 16000 -ac 1 chunk.wav
```

The `-ss` flag placed **before** `-i` triggers ffmpeg's keyframe seek, making
extraction time proportional to the chunk length rather than the full file.

The `audio_chunks()` context manager:
1. Computes the number of chunks from `ceil(duration / chunk_size)`.
2. Yields `(chunk_path, offset)` one at a time.
3. **Deletes each temp file** immediately after the caller yields control back,
   keeping peak disk usage to one chunk at a time.

### Timestamp Continuity

Each segment returned by faster-whisper has `start` and `end` times relative
to the start of the chunk wav file.  The `transcribe_chunk()` function adds
`time_offset` (the absolute start position of the chunk) to both fields,
producing timestamps that are unambiguously relative to the original file.

### Chunk Size Selection

Default: **1800 s (30 minutes)**.

| chunk_size | VRAM pressure | Overhead per chunk | Notes             |
|------------|---------------|---------------------|-------------------|
| 600 s      | Low           | Higher (more ops)   | Safe for 8 GB GPU |
| 1800 s     | Medium        | Balanced            | **Default**       |
| 3600 s     | Higher        | Minimal             | RTX 5000 Ada OK   |

Adjustable via `--chunk-size`.

---

## GPU Usage and Optimisation

### Model: `large-v3-turbo`

faster-whisper's `large-v3-turbo` is a distilled variant of `large-v3`.  It
achieves ~3× higher throughput at minimal accuracy loss.  On an RTX 5000 Ada
(32 GB VRAM) with `float16`, it processes audio at ~15–25× real-time speed.

### Compute type: `float16`

| Compute type      | VRAM     | Speed   | Accuracy |
|-------------------|----------|---------|----------|
| `float32`         | ~12 GB   | Slow    | Baseline |
| `float16`         | ~6 GB    | Fast    | ≈ fp32   |
| `int8_float16`    | ~4 GB    | Faster  | ≈ fp16   |
| `int8`            | ~3 GB    | Fastest | ↓ slight |

`float16` is the default: it maximises accuracy while using far less VRAM than
`float32`, leaving headroom on the 32 GB card for other processes.

### VAD Filtering

Voice Activity Detection is applied before Whisper inference by faster-whisper
using [silero-vad](https://github.com/snakers4/silero-vad).  It strips silence
and non-speech segments, reducing the number of tokens the encoder must process
and improving both speed and accuracy.  Parameters are tuned for law-enforcement
recordings with variable noise levels.

### Beam search

`beam_size=5, best_of=1, temperature=0.0`:
- Temperature = 0 → greedy/deterministic decoding (fastest, no sampling).
- `best_of=1` → no fallback sampling (saves time).
- `beam_size=5` → moderate accuracy improvement over greedy at low cost.

### `condition_on_previous_text=True`

The Whisper decoder is conditioned on the previous segment's text.  This
maintains contextual coherence across segments (important for technical
law-enforcement terminology) and reduces hallucinations.

---

## Error Handling Strategy

The project uses a layered error strategy:

| Layer         | Approach                                              |
|---------------|-------------------------------------------------------|
| `audio.py`    | `RuntimeError` with the ffmpeg stderr for diagnosis   |
| `transcriber` | Exceptions propagate up; not caught at this level     |
| `main.py`     | `try/except Exception` per file; logs and continues   |
| CLI           | `sys.exit(1)` only for non-existent root path         |

The key invariant: **one bad file never stops processing of subsequent files**.
The summary at the end lists all failures with their error messages.

Transient ffmpeg errors (e.g., malformed atoms in partially-downloaded files)
produce a `RuntimeError` with the full stderr included, making diagnosis
straightforward without needing to re-run ffmpeg manually.

---

## Skip / Resume Logic

A file is skipped when **both** output files (`.txt` and `.srt`) exist **and**
both have a modification time strictly newer than the source media file.

```python
source_mtime < min(txt_mtime, srt_mtime)
```

This means:
- A run interrupted mid-file will reprocess that file (outputs are either
  missing or older than the source).
- A completed file is never reprocessed unless the source changes.
- Files whose source is updated after transcription will be re-queued
  automatically.

---

## Data Flow

```
Filesystem → scanner.py → [Path, ...]
                              │
                              ▼
                          audio.py
                    ffprobe → duration: float
                    ffmpeg  → (chunk_path: Path, offset: float) *
                              │
                              ▼
                        transcriber.py
                    WhisperModel.transcribe() → [Segment, ...]
                    apply time_offset          → absolute timestamps
                              │
                              ▼
                          writer.py
                    write_txt() → .txt
                    write_srt() → .srt
```

All inter-module communication uses plain Python types (`Path`, `float`,
`List[Segment]`) — no shared state, no global variables, no side-channel
coupling.

---

## Key Design Decisions and Trade-offs

### 1. Subprocess ffmpeg over Python bindings

**Decision**: Use `subprocess.run(["ffmpeg", ...])` rather than `ffmpeg-python`
or `pydub`.

**Rationale**: Direct subprocess calls are explicit, debuggable (stderr is
captured and surfaced in exceptions), have zero additional dependencies, and
allow the full range of ffmpeg flags.  `ffmpeg-python` would add a dependency
for no functional gain at this level of usage.

### 2. Temp files over in-memory numpy arrays

**Decision**: Each chunk is written to a temp WAV file, not loaded into a numpy
array.

**Rationale**: For 4-hour files chunked at 30 minutes, peak in-memory audio is
~60 MB (one chunk).  The alternative — loading all audio into a numpy array —
would require ~460 MB for a 4-hour file before any model VRAM is allocated.
Temp-file approach also allows ffmpeg to do format conversion before the Python
process ever sees the audio.

### 3. Model loaded once per process

**Decision**: `load_model()` is called once in `run()` and passed through.

**Rationale**: Model loading takes ~5–15 seconds.  Loading it once amortises
that cost across all files.  For multi-worker mode (separate processes), each
worker loads its own copy — this is the only supported mode of true parallelism
given GPU memory constraints.

### 4. No async I/O

**Decision**: Sequential processing with `concurrent.futures.ProcessPoolExecutor`
for optional multi-worker mode.

**Rationale**: The bottleneck is GPU inference (single model, single GPU).
Adding async coroutines would not improve throughput and would significantly
complicate the code.  Multi-process workers are only beneficial with multiple
GPUs.

### 5. `condition_on_previous_text=True`

**Decision**: Enable context conditioning across segments within each chunk.

**Rationale**: Law-enforcement vocabulary (call signs, codes, names, acronyms)
benefits greatly from contextual conditioning.  Without it, the model may
transcribe "10-30" inconsistently.  The minor risk of hallucination propagation
is outweighed by the terminology preservation benefit.

---

## XP and Clean Code Principles Applied

### Single Responsibility

Every module has exactly one reason to change:
- `scanner.py` changes only if skip logic changes.
- `audio.py` changes only if the ffmpeg strategy changes.
- `transcriber.py` changes only if the Whisper API changes.
- `writer.py` changes only if the output format changes.

### No Duplication (DRY)

- The SRT timestamp formatter (`_format_srt_timestamp`) exists exactly once.
- ffmpeg invocation logic is in `_run_ffmpeg()` called by both `extract_audio`
  and `extract_audio_segment`.
- All constants are in `config.py`; nothing is hardcoded elsewhere.

### Expressive Naming

Function names are verb phrases describing what they do:
`transcribe_chunk`, `extract_audio_segment`, `iter_media_files`,
`write_srt`, `load_model`, `_already_processed`.

### Configuration Separated from Logic

`TranscriptionConfig` is a `dataclass` populated at parse time and passed
explicitly through the call stack.  No module reads from global state or
environment variables directly.

### Small Functions

No function exceeds ~50 lines.  Each does one thing and is independently
testable.

### Tests for Pure Logic

The test suite covers all pure-logic components (writer, scanner, config)
without requiring a GPU, ffmpeg, or model download.

---

## Scalability Considerations

### Vertical (bigger files)

The chunking strategy scales linearly with file duration.  A 4-hour file
produces 8 × 30-minute chunks.  Processing time is `O(duration)`;  memory
usage is `O(chunk_size)`.

### Horizontal (more files, more GPUs)

Set `--workers N` where N equals the number of available GPUs.  Each
`ProcessPoolExecutor` worker is a separate Python process loading its own model
instance onto its assigned GPU.  CUDA device affinity can be set via the
`CUDA_VISIBLE_DEVICES` environment variable before launching:

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py /recordings --workers 2
```

### Throughput Estimation

On RTX 5000 Ada (32 GB VRAM) with `large-v3-turbo` + `float16`:
- Real-time factor (RTF) ≈ 0.04–0.07 (i.e., 15–25× faster than real-time)
- A 4-hour file ≈ 10–16 minutes transcription time
- A 1,000-file backlog of 1-hour recordings ≈ 3–5 hours total
