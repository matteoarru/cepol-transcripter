# CEPOL Transcripter

Production-ready batch transcription tool for law enforcement audio and video recordings.
Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with full CUDA acceleration.

---

## Features

- Recursively scans a root folder for all audio and video files
- GPU-accelerated transcription via faster-whisper (CTranslate2 backend)
- Handles files up to 4+ hours without loading them into memory (ffmpeg-based chunking)
- For long audio and video files, pre-extracts future chunks in background threads while the current chunk is transcribed
- Produces `.txt` plain-text transcripts and `.srt` subtitle files
- Saves outputs next to each source file, keeping the same basename and changing only the extension
- Logs the exact saved output paths for each successful file
- Skip logic: re-runs only files whose outputs are missing or stale
- Configurable chunking, VAD filtering, model, and parallelism
- Law-enforcement optimised: English only, preserves casing, punctuation, and technical terms

---

## Supported file types

| Category | Extensions              |
|----------|-------------------------|
| Audio    | `.wav` `.mp3` `.m4a` `.flac` |
| Video    | `.mp4` `.mkv` `.avi` `.mov`  |

---

## Requirements

| Dependency | Version  | Notes                      |
|------------|----------|----------------------------|
| Python     | ≥ 3.10   |                            |
| ffmpeg     | any      | Must be in `PATH`          |
| CUDA       | ≥ 11.8   | Required for GPU inference |
| cuDNN      | ≥ 8.x    | Required by CTranslate2    |

---

## Installation

### 1. Install system dependencies

```bash
# ffmpeg
sudo apt update && sudo apt install -y ffmpeg

# CUDA toolkit (if not already installed)
# Follow: https://developer.nvidia.com/cuda-downloads
```

### 2. Clone the repository

```bash
git clone <repo-url>
cd cepol-transcripter
```

### 3. Create the virtual environment and install Python packages

```bash
bash setup.sh
```

This script will:
- Detect your Python version (3.10+ required)
- Create `./venv/`
- Install all packages from `requirements.txt`

### 4. Activate the environment

```bash
source venv/bin/activate
```

---

## Usage

### Basic usage

```bash
python main.py /path/to/recordings
```

### With options

```bash
python main.py /path/to/recordings \
    --model large-v3-turbo \
    --chunk-size 1200 \
    --workers 1 \
    --pipeline-threads 5 \
    --log-level INFO
```

### Using Make

```bash
make setup
make run FOLDER=/path/to/recordings
make test
```

---

## CLI options

| Flag              | Default           | Description                                          |
|-------------------|-------------------|------------------------------------------------------|
| `root`            | (required)        | Root folder to scan recursively                      |
| `--model`         | `large-v3-turbo`  | Whisper model name (see model table below)           |
| `--workers`       | `1`               | Parallel workers (use 1 for single-GPU setups)       |
| `--pipeline-threads` | `5`            | Background chunk-extraction threads for long media   |
| `--chunk-size`    | `1200`            | Max chunk length in seconds (default = 20 min)       |
| `--no-vad`        | VAD enabled       | Disable Voice Activity Detection filtering           |
| `--max-duration`  | unlimited         | Cap processed duration per file (useful for testing) |
| `--device`        | `cuda`            | `cuda` or `cpu`                                      |
| `--compute-type`  | `float16`         | `float16`, `int8_float16`, `int8`, `float32`         |
| `--log-level`     | `INFO`            | `DEBUG`, `INFO`, `WARNING`, `ERROR`                  |

### `.env` tuning

You can set defaults in `.env`:

```dotenv
MEDIA_CHUNK_MINUTES=20
PIPELINE_THREADS=5
```

- `MEDIA_CHUNK_MINUTES` controls the default chunk slot size for long media.
- `PIPELINE_THREADS` controls how many background ffmpeg extraction threads are used for long media.
- CLI flags still override the `.env` defaults.

## Code Structure

- `main.py`: CLI entry point and top-level orchestration.
- `src/processor.py`: single-file processing workflow.
- `src/audio.py`: ffprobe/ffmpeg helpers, chunk planning, and pipelined extraction.
- `src/transcriber.py`: faster-whisper integration and transcript segment DTOs.
- `src/writer.py`: `.txt` and `.srt` formatting.
- `src/scanner.py`: recursive media discovery and skip logic.
- `src/output_paths.py`: single source of truth for sibling output filenames.

## Development Style

The codebase follows a simple clean-code and XP-friendly workflow:

- Keep modules focused on one responsibility.
- Prefer small refactors backed by tests over large rewrites.
- Preserve one clear source of truth for shared rules such as config and output paths.
- Verify changes with `venv/bin/python -m pytest -q` and `venv/bin/python -m pyright`.

### Model options (speed vs accuracy trade-off)

| Model             | VRAM   | Speed  | Accuracy |
|-------------------|--------|--------|----------|
| `tiny`            | ~1 GB  | ★★★★★ | ★★☆☆☆   |
| `base`            | ~1 GB  | ★★★★☆ | ★★★☆☆   |
| `small`           | ~2 GB  | ★★★★☆ | ★★★☆☆   |
| `medium`          | ~5 GB  | ★★★☆☆ | ★★★★☆   |
| `large-v3-turbo`  | ~6 GB  | ★★★★☆ | ★★★★★   |
| `large-v3`        | ~10 GB | ★★☆☆☆ | ★★★★★   |

`large-v3-turbo` is the recommended default: it offers near large-v3 accuracy at ~3× the speed.

---

## Example outputs

### `.txt` transcript

```
Officer Martinez, can you describe the sequence of events?

At approximately 21:40 hours we received a dispatch call for a 10-30
at the intersection of Fifth Avenue and Monroe Street.
The suspect vehicle, a dark blue Ford Transit, was travelling northbound
at an estimated 70 miles per hour.

Forensics later confirmed the VIN matched a stolen vehicle report
filed in District 4 on the 14th.
```

### `.srt` subtitle file

For example, `/evidence/case-7/interview.mp4` produces:
- `/evidence/case-7/interview.txt`
- `/evidence/case-7/interview.srt`

```
1
00:00:03,240 --> 00:00:06,880
Officer Martinez, can you describe the sequence of events?

2
00:00:08,120 --> 00:00:14,560
At approximately 21:40 hours we received a dispatch call for a 10-30
at the intersection of Fifth Avenue and Monroe Street.

3
00:00:15,040 --> 00:00:21,800
The suspect vehicle, a dark blue Ford Transit, was travelling northbound
at an estimated 70 miles per hour.
```

---

## Troubleshooting

### `CUDA not available` / falls back to CPU

- Verify your CUDA installation: `nvidia-smi`
- Check CTranslate2 sees the GPU: `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"`
- Reinstall with CUDA support: `pip install ctranslate2 --extra-index-url https://pypi.nvidia.com`

### `ffmpeg: command not found`

```bash
sudo apt install ffmpeg
```

### Model download fails / slow

faster-whisper downloads models from Hugging Face on first use.
To pre-download manually:
```python
from faster_whisper import WhisperModel
WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
```

### Out of GPU memory

- Reduce `--chunk-size` to process shorter segments at a time
- Switch to `--compute-type int8_float16` or `--compute-type int8`
- Use a smaller model: `--model medium`

### Very slow on CPU

Set `--device cuda`. If CUDA is unavailable, set `--compute-type int8` for better CPU performance.

### Corrupted file skipped silently

Enable debug logging: `--log-level DEBUG`. Each failed file is logged with the full exception.
