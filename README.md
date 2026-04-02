# CEPOL Transcripter

Production-ready batch transcription tool for law enforcement audio and video recordings.
Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with full CUDA acceleration.

Current release: `v2.0.0`

`v2.0.0` focuses on high-throughput batch execution:

- pipelined extraction for long audio and video files
- worker-local model reuse and one-time CUDA startup probing
- benchmark-backed throughput presets for large transcription batches

---

## Features

- Recursively scans a root folder for all audio and video files
- GPU-accelerated transcription via faster-whisper (CTranslate2 backend)
- Handles files up to 4+ hours without loading them into memory (ffmpeg-based chunking)
- For long audio and video files, pre-extracts future chunks in background threads while the current chunk is transcribed
- Adapts chunk source strategy: reuses cached `*.audio.wav` sidecars when available and falls back to direct chunk extraction when that is faster
- Probes CUDA once at startup and falls back to CPU immediately if the runtime is broken
- Reuses one loaded model per worker process instead of reloading per file
- Produces `.txt` plain-text transcripts and `.srt` subtitle files
- Saves outputs next to each source file, keeping the same basename and changing only the extension
- Logs the exact saved output paths for each successful file
- Logs stage-level timing breakdowns for probing, preparation, extraction, transcription, and writing
- Skip logic: does not reprocess files when both sibling `.txt` and `.srt` outputs already exist
- Removes generated sibling `*.audio.wav` files after a successful transcription run
- Configurable chunking, VAD filtering, model, and parallelism
- Law-enforcement optimised: English only, preserves casing, punctuation, and technical terms

---

## Supported file types

| Category | Extensions              |
|----------|-------------------------|
| Audio    | `.wav` `.mp3` `.m4a` `.flac` |
| Video    | `.mp4` `.mkv` `.avi` `.mov`  |

Extension matching is case-insensitive, so files such as `INTERVIEW.MP4` and
`audio.WAV` are processed normally.

---

## Requirements

| Dependency | Version  | Notes                                    |
|------------|----------|------------------------------------------|
| Python     | ≥ 3.10   |                                          |
| ffmpeg     | any      | Must be in `PATH`                        |
| CUDA       | 12.x     | Required for current GPU faster-whisper  |
| cuDNN      | 9.x      | Required for current GPU faster-whisper  |

---

## Installation

### 1. Install system dependencies

```bash
sudo apt update
sudo add-apt-repository -y multiverse
sudo apt install -y ffmpeg libcublas12 libcublaslt12
```

For current `faster-whisper` / `ctranslate2` GPU builds you also need cuDNN 9.
This project supports three practical CUDA setups:

1. Ubuntu packages for ffmpeg + cuBLAS, and NVIDIA package-manager install for cuDNN 9.
2. Ubuntu packages for ffmpeg + cuBLAS, and Python wheels for cuDNN/cuBLAS in the venv.
3. CPU-only mode, which requires only `ffmpeg`.

### 1a. Ubuntu 24.04 system-level cuDNN 9 install

If you want a pure OS-level CUDA runtime, follow NVIDIA's package-manager path
for Ubuntu 24.04 and CUDA 12:

```bash
sudo apt-get install -y zlib1g
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cudnn9-cuda-12
```

Optional verification package:

```bash
sudo apt-get install -y libcudnn9-samples
```

If you install cuDNN this way, the app can use the system runtime directly.

### 1b. Python-wheel CUDA runtime inside the venv

If you prefer not to manage cuDNN system-wide, the app also supports the
official Linux Python wheels recommended by `faster-whisper`:

```bash
venv/bin/pip install --only-binary=:all: \
    nvidia-cublas-cu12==12.9.1.4 \
    nvidia-cudnn-cu12==9.20.0.48
```

This project auto-detects and preloads those Python-packaged CUDA libraries on
startup, so you do not need to export `LD_LIBRARY_PATH` manually.

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

### Launcher scripts

Linux / macOS:

```bash
./run.sh /path/to/recordings
```

Windows:

```bat
run.bat C:\path\to\recordings
```

Both launchers use the project's virtual environment and forward any extra CLI flags:

```bash
./run.sh sample_media --log-level INFO
./run.sh --version
```

### With options

```bash
python main.py /path/to/recordings \
    --model large-v3-turbo \
    --chunk-size 1200 \
    --workers 1 \
    --pipeline-threads 5 \
    --beam-size 1 \
    --compute-type int8_float16 \
    --no-condition-on-previous-text \
    --no-vad \
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
| `--version`       |                   | Print the application version and exit               |
| `--model`         | `large-v3-turbo`  | Whisper model name (see model table below)           |
| `--workers`       | `1`               | Parallel workers; capped to visible GPUs in CUDA mode |
| `--cpu-threads`   | `8`               | CTranslate2 CPU threads                              |
| `--pipeline-threads` | `5`            | Background chunk-extraction threads for long media   |
| `--chunk-size`    | `1200`            | Max chunk length in seconds (default = 20 min)       |
| `--no-vad`        | VAD enabled       | Disable Voice Activity Detection filtering           |
| `--beam-size`     | `5`               | Decoding beam size                                   |
| `--best-of`       | `1`               | Candidate count for sampling-based decoding          |
| `--no-condition-on-previous-text` | enabled | Disable cross-chunk text conditioning         |
| `--max-duration`  | unlimited         | Cap processed duration per file (useful for testing) |
| `--device`        | `cuda`            | `cuda` or `cpu`                                      |
| `--compute-type`  | `float16`         | `float16`, `int8_float16`, `int8`, `float32`         |
| `--log-level`     | `INFO`            | `DEBUG`, `INFO`, `WARNING`, `ERROR`                  |

### Performance tuning guide

- Single GPU: keep `--workers 1`.
- Multiple GPUs: increase `--workers` up to the number of visible GPUs. The app pins one worker to one GPU.
- CPU runs: use `--device cpu --compute-type int8`, then tune `--cpu-threads` and optionally `--workers`.
- Batch throughput: use `--compute-type int8_float16 --beam-size 1 --no-condition-on-previous-text`.
- Dense speech / meeting-style recordings: `--no-vad` can improve throughput further.
- Better continuity across chunks: keep `--condition-on-previous-text` enabled.
- Sparse or noisy recordings: keep VAD enabled unless benchmarks on your own material show otherwise.
- Long files: keep `--pipeline-threads > 1` so extraction overlaps with transcription.

### `.env` tuning

The app supports the main throughput knobs directly from `.env`.
For a batch-throughput profile, use:

```dotenv
MEDIA_CHUNK_MINUTES=20
CPU_THREADS=8
PIPELINE_THREADS=5
WHISPER_COMPUTE_TYPE=int8_float16
WHISPER_VAD=false
WHISPER_BEAM_SIZE=1
WHISPER_BEST_OF=1
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
```

- `MEDIA_CHUNK_MINUTES` controls the default chunk slot size for long media.
- `CPU_THREADS` controls CTranslate2 host-side CPU parallelism.
- `PIPELINE_THREADS` controls how many background ffmpeg extraction threads are used for long media.
- `WHISPER_COMPUTE_TYPE` controls the default quantisation mode for inference.
- `WHISPER_VAD=false` disables Voice Activity Detection by default.
- `WHISPER_BEAM_SIZE` and `WHISPER_BEST_OF` let you trade quality for speed.
- `WHISPER_CONDITION_ON_PREVIOUS_TEXT=false` can improve speed and reduce cross-chunk error carry-over.
- CLI flags still override the `.env` defaults.

For a more conservative quality-first profile, set:

```dotenv
WHISPER_COMPUTE_TYPE=float16
WHISPER_VAD=true
WHISPER_BEAM_SIZE=5
WHISPER_CONDITION_ON_PREVIOUS_TEXT=true
```

## Benchmarks

The benchmark figures below were captured locally on April 1, 2026 on:

- Ubuntu 24.04.4 LTS
- NVIDIA RTX 5000 Ada Generation (32 GB VRAM)
- NVIDIA driver `570.211.01`, CUDA `12.8`
- `ffmpeg 6.1.1-3ubuntu5`
- `large-v3-turbo`, single GPU, `--workers 1`, `--pipeline-threads 5`

Methodology:

- two isolated copies of the provided 74.9-minute sample video were processed per run
- total processed duration per run was about 149.8 minutes
- Hugging Face model downloads were already cached
- isolated temporary roots were used so skip logic did not affect timings

### Benchmark results

| Configuration | Wall time | Throughput | Notes |
|---------------|-----------|------------|-------|
| `--compute-type int8_float16 --beam-size 1 --no-condition-on-previous-text --no-vad` | `54.4s` | `165.2x` real time | fastest measured on this sample |
| `--beam-size 1 --no-condition-on-previous-text --no-vad` | `59.2s` | `151.8x` real time | best `float16` result |
| `--compute-type int8_float16 --beam-size 1 --no-condition-on-previous-text` | `65.1s` | `138.1x` real time | safer if you want to keep VAD |
| `--beam-size 1 --no-condition-on-previous-text` | `72.4s` | `124.2x` real time | largest single tuning win vs baseline |
| baseline defaults | `90.0s` | `99.9x` real time | previous quality-first tuning |

Key findings:

- Cross-chunk conditioning was the biggest throughput cost on this sample.
- `int8_float16` outperformed `float16` once decoding was simplified.
- `--no-vad` helped on this dense-speech recording, but it may be worse on sparse or noisy material.
- Extraction stayed around `5.7-6.1s` in all runs, so transcription remained the dominant bottleneck.

### Recommended profiles

Fastest measured profile:

```bash
python main.py /path/to/recordings \
    --device cuda \
    --compute-type int8_float16 \
    --beam-size 1 \
    --no-condition-on-previous-text \
    --no-vad
```

Safer throughput profile for mixed real-world batches:

```bash
python main.py /path/to/recordings \
    --device cuda \
    --compute-type int8_float16 \
    --beam-size 1 \
    --no-condition-on-previous-text
```

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
- Check the required userspace libraries exist:
  - `ldconfig -p | rg 'libcublas|libcudnn'`
  - for current faster-whisper builds you need `libcublas.so.12` and cuDNN 9
- On Ubuntu 24.04, install the distro cuBLAS packages:

```bash
sudo add-apt-repository -y multiverse
sudo apt update
sudo apt install -y libcublas12 libcublaslt12
```

- Then install cuDNN 9 from NVIDIA's package-manager instructions:

```bash
sudo apt-get install -y zlib1g
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cudnn9-cuda-12
```

- Or use the supported Linux `pip` runtime libraries instead:

```bash
venv/bin/pip install --only-binary=:all: \
    nvidia-cublas-cu12==12.9.1.4 \
    nvidia-cudnn-cu12==9.20.0.48
```

- Quick runtime check:

```bash
venv/bin/python - <<'PY'
from src.transcriber import missing_cuda_runtime_libraries
print(missing_cuda_runtime_libraries())
PY
```

An empty tuple `()` means the required CUDA runtime libraries are available.

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

Set `--device cuda`. If CUDA is unavailable:

- use `--compute-type int8`
- increase `--cpu-threads`
- use `--workers > 1` for multi-file CPU batches
- lower `--beam-size` to `1` or `2` when speed matters more than accuracy

### Corrupted file skipped silently

Enable debug logging: `--log-level DEBUG`. Each failed file is logged with the full exception.
