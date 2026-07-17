# Whisper Patois Fine-Tune — Training Manual

> **Goal**: Fine-tune a Whisper model to transcribe Jamaican Patois audio into plain English text.
>
> **Approach**: Self-training with pseudo-labeling — generate initial transcripts with a base Whisper model, manually correct them to accurate English translations, then fine-tune on the corrected pairs.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Project Structure](#2-project-structure)
3. [Setup & Installation](#3-setup--installation)
4. [Data Preparation (The Critical Step)](#4-data-preparation-the-critical-step)
5. [Training](#5-training)
6. [Evaluation](#6-evaluation)
7. [Inference — Using the Model](#7-inference--using-the-model)
8. [Delivery Options](#8-delivery-options)
9. [The Data Flywheel](#9-the-data-flywheel)
10. [Reference Benchmarks](#10-reference-benchmarks)
11. [Troubleshooting](#11-troubleshooting)
12. [Frequently Asked Questions](#12-frequently-asked-questions)

---

## 1. System Requirements

### Hardware

| Component | Minimum                    | Recommended                               |
| --------- | -------------------------- | ----------------------------------------- |
| GPU       | 8 GB VRAM (e.g., RTX 3070) | 16+ GB VRAM (e.g., RTX 4090, A10, A100)   |
| RAM       | 16 GB                      | 32+ GB                                    |
| Storage   | 50 GB free                 | 100+ GB free (for datasets + checkpoints) |
| CPU       | 4 cores                    | 8+ cores                                  |

**GPU memory estimates by model size and training mode:**

| Model              | Full Fine-Tune | LoRA  |
| ------------------ | -------------- | ----- |
| `whisper-small`    | ~6 GB          | ~4 GB |
| `whisper-medium`   | ~10 GB         | ~6 GB |
| `whisper-large-v3` | ~16 GB         | ~8 GB |

> **No GPU?** You can still run pseudo-label generation and inference on CPU, but it will be very slow. Training requires a GPU.

### Software

- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows (WSL2 recommended)
- **Python**: 3.8–3.11
- **CUDA**: 11.8 or 12.x (if using NVIDIA GPU)

---

## 2. Project Structure

```
patois_whisper_finetune/
├── data/
│   ├── raw/                # ← Place your original audio files here
│   ├── processed/          # Resampled 16kHz mono WAV files (auto-generated)
│   ├── transcripts/        # ← Place your corrected transcript files here
│   └── dataset/            # Hugging Face Dataset artifacts (auto-generated)
├── scripts/
│   ├── prepare_data.py     # Audio resampling + pseudo-labels + dataset building
│   ├── train.py            # Fine-tuning (standard full fine-tune + LoRA)
│   ├── evaluate.py         # WER/CER evaluation on a test set
│   └── transcribe.py       # Inference with the fine-tuned model
├── models/
│   └── checkpoints/        # Training checkpoints and final model (auto-generated)
├── config.yaml             # Hyperparameters
├── requirements.txt        # Python dependencies
├── setup_environment.sh    # One-click environment setup
└── README.md               # Quick-start guide
```

---

## 3. Setup & Installation

### 3.1 Automated Setup (Recommended)

```bash
cd patois_whisper_finetune
bash setup_environment.sh
source venv/bin/activate
```

This script will:

1. Create a Python virtual environment (`venv/`)
2. Detect your CUDA version and install the correct PyTorch build
3. Install all dependencies from `requirements.txt`

### 3.2 Manual Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install PyTorch (choose the right CUDA version)
# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### 3.3 Verify Installation

```bash
source venv/bin/activate

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check all scripts parse correctly
python -c "
import ast
for s in ['scripts/prepare_data.py', 'scripts/train.py', 'scripts/evaluate.py', 'scripts/transcribe.py']:
    ast.parse(open(s).read())
    print(f'{s}: OK')
"

# Check config loads
python -c "
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'Config OK — model: {cfg[\"model\"][\"name_or_path\"]}')
"
```

**Expected output:**

```
PyTorch 2.x.x
CUDA available: True
scripts/prepare_data.py: OK
scripts/train.py: OK
scripts/evaluate.py: OK
scripts/transcribe.py: OK
Config OK — model: openai/whisper-medium
```

---

## 4. Data Preparation (The Critical Step)

This is the most important and time-consuming part. The quality of your transcripts directly determines the quality of your final model.

> **Heuristic**: 1 minute of audio takes approximately 5–10 minutes to manually correct for an experienced listener. Plan your effort accordingly.

### 4.1 End-to-End Flow

```
┌─────────────────────────────────────────────────────────┐
│  1. Place audio files in data/raw/                      │
│  2. Run prepare_data.py to resample + generate labels   │
│  3. Manually correct transcripts in data/transcripts/   │
│  4. Re-run prepare_data.py --skip_pseudo to build       │
│     the dataset                                         │
│  5. Train!                                              │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Step 1: Organize Audio

Place all your audio files in `data/raw/`. Supported formats: WAV, MP3, FLAC, M4A, OGG, Opus. Subdirectories are supported — files are found recursively.

> **Tip**: Use audio with clear speech, minimal background noise, and consistent volume levels.

### 4.3 Step 2: Generate Pseudo-Labels (First Pass)

```bash
python scripts/prepare_data.py --raw_dir data/raw --model_size medium
```

**What happens:**

1. All audio files are resampled to 16 kHz mono WAV and saved to `data/processed/`
2. The base Whisper-medium model is downloaded from Hugging Face
3. Each audio file is transcribed automatically
4. The transcripts are printed to the terminal
5. The script exits with instructions to correct the transcripts

**Flags:**

| Flag           | Default          | Description                                                                                            |
| -------------- | ---------------- | ------------------------------------------------------------------------------------------------------ |
| `--raw_dir`    | `data/raw`       | Directory with source audio files                                                                      |
| `--model_size` | `medium`         | Whisper model for pseudo-labeling (`tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`) |
| `--batch_size` | `8`              | Batch size for GPU inference                                                                           |
| `--output_dir` | `data/processed` | Where resampled WAV files are saved                                                                    |

**Common issues:**

- **OOM**: Reduce `--batch_size` to 4 or 2
- **Slow on CPU**: Expected. Use a GPU or cloud instance temporarily for this step.

### 4.4 Step 3: Correct Transcripts

The pseudo-labels will be imperfect. Your job is to correct each transcript to **accurate English text**.

**Expected quality of pseudo-labels:** Untrained Whisper-medium on Patois has ~89% WER. Don't be discouraged — even bad pseudo-labels save you from transcribing from scratch.

**Transcript format options:**

**JSON** (recommended for small/medium datasets) — save one or more `.json` files in `data/transcripts/`:

```json
{
  "audio_file_001": "The corrected English transcript goes here.",
  "audio_file_002": "Another corrected transcript.",
  "audio_file_003": "Make sure the filename stems match exactly."
}
```

**CSV** (good for spreadsheet editing):

```csv
filename,transcript
audio_file_001,The corrected English transcript goes here.
audio_file_002,Another corrected transcript.
```

**IMPORTANT:**

- Filename stems (without extension) must match exactly — `data/processed/my_clip.wav` needs key `my_clip`
- Do NOT include the `.wav` extension in the transcript key
- All text should be **plain English** — not Patois orthography. The goal is English output.
- Keep punctuation simple — periods and commas are fine. Avoid special characters.

### 4.5 Step 4: Build the Dataset (Second Pass)

```bash
python scripts/prepare_data.py --skip_pseudo
```

**What happens:**

1. Loads your corrected transcripts from `data/transcripts/`
2. Matches them to the resampled audio files in `data/processed/`
3. Creates an 80/10/10 train/validation/test split
4. Saves the dataset to `data/dataset/`
5. Saves `data/dataset/dataset_info.json` with statistics

**Additional flags:**

| Flag            | Default        | Description                     |
| --------------- | -------------- | ------------------------------- |
| `--val_split`   | `0.1`          | Fraction of data for validation |
| `--test_split`  | `0.1`          | Fraction of data for test       |
| `--dataset_dir` | `data/dataset` | Where to save the HF dataset    |

**Expected output:**

```
DATASET READY FOR TRAINING
  Train:      340 samples
  Validation: 43 samples
  Test:       43 samples
```

**Troubleshooting "No matching pairs found":**

```
ValueError: No matching (audio, transcript) pairs found.
```

→ Check transcript keys match audio stems exactly (no extension)
→ Verify `data/processed/` exists and has `.wav` files
→ Run from the project root directory

---

## 5. Training

### 5.1 Configuration (`config.yaml`)

Key settings:

| Setting                                | Default                 | Notes                                           |
| -------------------------------------- | ----------------------- | ----------------------------------------------- |
| `model.name_or_path`                   | `openai/whisper-medium` | Change to `openai/whisper-small` for lower VRAM |
| `training.num_steps`                   | `4000`                  | Total training steps                            |
| `training.learning_rate`               | `1.0e-5`                | Peak learning rate                              |
| `training.per_device_train_batch_size` | `8`                     | Reduce to 4 or 2 if OOM                         |
| `lora.enabled`                         | `false`                 | Toggle with `--use_lora` flag                   |
| `data.max_audio_length`                | `30.0`                  | Trim audio longer than this (seconds)           |

**Recommended settings by dataset size:**

| Dataset Size | Model      | Mode | Steps | LR   |
| ------------ | ---------- | ---- | ----- | ---- |
| < 5 hours    | `small`    | LoRA | 2000  | 2e-5 |
| 5–20 hours   | `small`    | Full | 4000  | 1e-5 |
| 20–40 hours  | `medium`   | LoRA | 4000  | 1e-5 |
| 40+ hours    | `medium`   | Full | 4000  | 1e-5 |
| 40+ hours    | `large-v3` | LoRA | 4000  | 1e-5 |

### 5.2 Running Training

```bash
# Standard full fine-tune
python scripts/train.py --config config.yaml

# LoRA (faster, smaller output — 2× speedup, 40% less memory)
python scripts/train.py --config config.yaml --use_lora

# Resume from checkpoint (if interrupted)
python scripts/train.py --config config.yaml --resume models/checkpoints/checkpoint-1000
```

### 5.3 Monitoring with TensorBoard

```bash
tensorboard --logdir models/checkpoints/logs
# Open http://localhost:6006
```

Track **WER** (should decrease) and **loss** (should decrease steadily).

**Interpreting WER during training:**

| WER       | Meaning                                                              |
| --------- | -------------------------------------------------------------------- |
| > 0.80    | Barely better than untrained — may need more data or longer training |
| 0.50–0.80 | Improving — roughly every other word is correct                      |
| 0.30–0.50 | Good — most words are correct                                        |
| < 0.30    | Excellent — approaching human-level for this task                    |
| < 0.15    | Outstanding — high accuracy transcription                            |

### 5.4 Adjusting Training Mid-Run

- **WER plateaued early?** Stop with Ctrl+C — the best checkpoint is automatically loaded at the end.
- **Still improving at 4000 steps?** Increase `num_steps` in `config.yaml` and resume:
  ```bash
  python scripts/train.py --config config.yaml --resume models/checkpoints/checkpoint-4000
  ```
- **Loss is NaN?** Reduce `learning_rate` to 5e-6 in `config.yaml`.

### 5.5 Multi-GPU Training

The script automatically supports DataParallel when multiple GPUs are detected:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --config config.yaml
```

---

## 6. Evaluation

### 6.1 Basic Evaluation

```bash
python scripts/evaluate.py --model_path models/checkpoints --split test
```

**Output:**

```
EVALUATION RESULTS
  Samples:    43
  WER:        0.3245
  CER:        0.1521

Sample predictions (first 10):
  [0] Ref:  the man went to the store to buy some food
      Hyp:  the man went to the store to buy some food
```

### 6.2 All Flags

| Flag             | Default                          | Description                                       |
| ---------------- | -------------------------------- | ------------------------------------------------- |
| `--model_path`   | (required)                       | Path to the fine-tuned model                      |
| `--dataset_path` | `data/dataset`                   | Path to the prepared dataset                      |
| `--split`        | `test`                           | Split to evaluate (`train`, `validation`, `test`) |
| `--batch_size`   | `8`                              | Inference batch size                              |
| `--device`       | `auto`                           | Override device (`cuda` or `cpu`)                 |
| `--output`       | `models/evaluation_results.json` | Path to save results                              |

### 6.3 Understanding Results

Results are saved to `models/evaluation_results.json` with full error analysis:

```json
{
  "wer": 0.3245,
  "cer": 0.1521,
  "num_samples": 43,
  "error_analysis": [
    { "audio": "data/processed/sample_001.wav", "reference": "the man went to the store", "hypothesis": "the man went to the store" }
  ]
}
```

- **WER** (Word Error Rate): Percentage of words that differ. Lower is better.
- **CER** (Character Error Rate): Character-level error rate. Lower is better.
- **Error analysis**: Every prediction paired with its reference. Use this to find patterns in mistakes.

---

## 7. Inference — Using the Model

### 7.1 Basic Usage

```bash
# Transcribe a single file (auto-detects LoRA vs full model)
python scripts/transcribe.py --model_path models/checkpoints --audio path/to/audio.wav

# Save output to file
python scripts/transcribe.py --model_path models/checkpoints --audio audio.wav --output result.txt

# Force CPU or CUDA
python scripts/transcribe.py --model_path models/checkpoints --audio audio.wav --device cpu
```

### 7.2 All Flags

| Flag           | Default    | Description                                    |
| -------------- | ---------- | ---------------------------------------------- |
| `--model_path` | (required) | Path to the fine-tuned model                   |
| `--audio`      | (required) | Path to the audio file to transcribe           |
| `--output`     | `None`     | Save transcription to file (stdout if omitted) |
| `--device`     | `auto`     | Override device (`cuda` or `cpu`)              |

### 7.3 Integration with Pyannote Pipeline

To use the fine-tuned model with a pyannote + Whisper pipeline:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import json, os

model_path = "models/checkpoints"
processor = WhisperProcessor.from_pretrained(model_path)

if os.path.exists(f"{model_path}/adapter_config.json"):
    with open(f"{model_path}/adapter_config.json") as f:
        cfg = json.load(f)
    base_name = cfg.get("base_model_name_or_path", "openai/whisper-medium")
    model = WhisperForConditionalGeneration.from_pretrained(base_name)
    model = PeftModel.from_pretrained(model, model_path)
else:
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

model.to("cuda").eval()
```

---

## 8. Delivery Options

### Option A — Full Model Weights (~3 GB)

Deliver the entire `models/checkpoints/` directory. Client replaces the default Whisper model in their pipeline with no code changes.

```python
model = WhisperForConditionalGeneration.from_pretrained("path/to/models/checkpoints")
processor = WhisperProcessor.from_pretrained("path/to/models/checkpoints")
```

**Pros**: Self-contained, no extra libraries needed.  
**Cons**: Large file.

### Option B — LoRA Adapter Only (~5 MB)

Deliver only `adapter_config.json` and `adapter_model.safetensors`. Client loads base model + adapter.

```python
from peft import PeftModel
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
model = PeftModel.from_pretrained(base, "path/to/adapter")
processor = WhisperProcessor.from_pretrained("path/to/adapter")
```

**Pros**: Tiny file, fast download, easy to update iteratively.  
**Cons**: Client must have base Whisper model.

---

## 9. The Data Flywheel

A virtuous cycle for continuous improvement:

```
1. Deploy model → transcribe new, unlabeled audio
2. Human lightly corrects the transcripts
3. Add corrected data to the training set
4. Retrain → improved model
5. Go back to step 1 (the model keeps getting better)
```

```bash
# 1. Transcribe new audio with current model
python scripts/transcribe.py --model_path models/checkpoints --audio new_audio.wav --output new_transcript.txt

# 2. [Correct the output manually]

# 3. Add corrected transcript to data/transcripts/ and rebuild dataset
python scripts/prepare_data.py --skip_pseudo

# 4. Retrain (resume from last checkpoint)
python scripts/train.py --config config.yaml --resume models/checkpoints/checkpoint-4000
```

**Expected diminishing returns:**

| Iteration | New Data               | Expected WER |
| --------- | ---------------------- | ------------ |
| Baseline  | 0 hours                | ~0.89        |
| 2nd       | +10 hours              | ~0.55        |
| 3rd       | +20 hours (cumulative) | ~0.40        |
| 4th       | +40 hours (cumulative) | ~0.30        |
| 5th+      | +60+ hours             | ~0.20–0.25   |

---

## 10. Reference Benchmarks

### Proven Results

| Configuration                             | WER   |
| ----------------------------------------- | ----- |
| Untrained Whisper-medium on Patois        | ~0.89 |
| After 20h fine-tune (small)               | ~0.51 |
| After 40h fine-tune (small)               | ~0.40 |
| After 40h fine-tune (medium)              | ~0.30 |
| After 40h + iterative correction (medium) | ~0.25 |

### Training Time Estimates (RTX 4090)

| Model            | Mode | 4000 steps |
| ---------------- | ---- | ---------- |
| `whisper-small`  | Full | ~32 min    |
| `whisper-small`  | LoRA | ~16 min    |
| `whisper-medium` | Full | ~80 min    |
| `whisper-medium` | LoRA | ~40 min    |

### Hyperparameter Reference

| Parameter            | Value | Notes                                   |
| -------------------- | ----- | --------------------------------------- |
| Optimizer            | AdamW | Standard for transformer fine-tuning    |
| Learning rate        | 1e-5  | Reduce to 5e-6 if loss diverges         |
| Warmup steps         | 500   | Linear warmup from 0 to peak LR         |
| Total steps          | 4000  | Extend to 6000–8000 for larger datasets |
| Effective batch size | 16    | 8 per device × 2 gradient accumulation  |
| FP16                 | True  | Mixed precision halves memory usage     |
| Generation beams     | 5     | Beam search width for evaluation        |

---

## 11. Troubleshooting

### 11.1 Installation

| Problem                              | Solution                                                    |
| ------------------------------------ | ----------------------------------------------------------- |
| `pip install torch` fails            | Install from pytorch.org with `--index-url`                 |
| `librosa` fails on macOS             | `brew install libsndfile ffmpeg` then `pip install librosa` |
| `soundfile` import fails             | `sudo apt-get install libsndfile1` (Linux)                  |
| `torch.cuda.is_available()` is False | Check `nvidia-smi` and reinstall matching CUDA PyTorch      |

### 11.2 GPU / Memory

| Problem                         | Solution                                                                  |
| ------------------------------- | ------------------------------------------------------------------------- |
| CUDA OOM during training        | Reduce `batch_size` → 4 or 2, use `--use_lora`, switch to `whisper-small` |
| CUDA OOM during pseudo-labeling | Reduce `--batch_size` to 2                                                |
| Training very slow              | Check `nvidia-smi`, try LoRA or smaller model                             |

### 11.3 Data

| Problem                   | Solution                                                                            |
| ------------------------- | ----------------------------------------------------------------------------------- |
| "No matching pairs found" | Check transcript keys match audio stems exactly; verify `data/processed/` has files |
| Audio files too long      | Increase `max_audio_length` in config (uses more memory)                            |
| Resample failures         | Check for corrupt audio files in `data/raw/`                                        |

### 11.4 Training

| Problem                | Solution                                                                        |
| ---------------------- | ------------------------------------------------------------------------------- |
| Loss is NaN            | Reduce learning rate to 5e-6, check for empty transcripts                       |
| WER not improving      | Check transcript quality, increase dataset size, try LoRA (acts as regularizer) |
| Disk space running out | `save_total_limit: 3` keeps only last 3 checkpoints; manually delete old ones   |

### 11.5 Evaluation / Inference

| Problem                    | Solution                                                                           |
| -------------------------- | ---------------------------------------------------------------------------------- |
| Model outputs gibberish    | May have overfit — check evaluation WER. Try CPU inference to rule out CUDA issues |
| LoRA adapter not detected  | Check `adapter_config.json` exists in `--model_path`                               |
| Audio format not supported | Convert to WAV first: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`             |

### 11.6 Debugging Checklist

```
[ ] Is the virtual environment activated?          source venv/bin/activate
[ ] Is CUDA available?                             python -c "import torch; print(torch.cuda.is_available())"
[ ] Is the config valid?                           python -c "import yaml; yaml.safe_load(open('config.yaml'))"
[ ] Does the dataset exist?                        ls data/dataset/
[ ] Are transcripts in the right place?            ls data/transcripts/
[ ] Is there enough disk space?                    df -h .
[ ] Is there enough GPU memory?                    nvidia-smi
[ ] Are the audio files valid?                     ffprobe data/raw/your_file.mp3
```

### 11.7 Enabling Debug Logging

Edit the logging line at the top of any script:

```python
# Change this:
logging.basicConfig(level=logging.INFO, ...)
# To this:
logging.basicConfig(level=logging.DEBUG, ...)
```

---

## 12. Frequently Asked Questions

**Q: How much data do I need?**
A: Even 10 hours of corrected audio yields meaningful improvements. 40+ hours is the proven sweet spot for Patois (WER ~0.30 with Whisper-medium). Start with what you have and add more later.

**Q: How long does training take?**
A: On an RTX 4090 with Whisper-medium: ~80 minutes for 4000 steps. With LoRA: ~40 minutes.

**Q: Can I use the model on CPU after training?**
A: Yes. Evaluation and inference work on CPU (slow but functional). Training requires a GPU.

**Q: What if my transcripts aren't perfect?**
A: Quality matters more than quantity. A small perfectly-corrected dataset beats a large noisy one. Review and fix systematically.

**Q: Can I fine-tune on non-Patois languages?**
A: Yes, the codebase is language-agnostic. Change `model.language` in `config.yaml` and update the tokenizer language in the training script if needed.

**Q: What's the difference between standard and LoRA training?**
A: Standard training updates all model weights. LoRA trains only small adapter matrices (~0.5% of parameters). LoRA is ~2× faster, uses ~40% less memory, and produces tiny output files (~5 MB vs ~3 GB) — with nearly identical quality for most tasks.

**Q: How do I know when to stop training?**
A: Watch the validation WER. When it stops improving for 500+ steps, you can stop. The best model is automatically loaded at the end.

**Q: Can I train on multiple GPUs?**
A: Yes. The script uses Hugging Face's `Seq2SeqTrainer` which supports DataParallel automatically.

**Q: I get "CUDA out of memory" even with batch size 1. What can I do?**
A: Try LoRA mode, switch to `whisper-small`, reduce `max_audio_length`, or add `gradient_checkpointing: true` to the config.

---

## Appendix: Quick Reference Card

```bash
# ── One-time setup ──────────────────────────────
bash setup_environment.sh && source venv/bin/activate

# ── Data pipeline ───────────────────────────────
python scripts/prepare_data.py --raw_dir data/raw --model_size medium
#   → [Manually correct transcripts in data/transcripts/]
python scripts/prepare_data.py --skip_pseudo

# ── Training ────────────────────────────────────
python scripts/train.py --config config.yaml               # Standard
python scripts/train.py --config config.yaml --use_lora     # LoRA

# ── Evaluation ──────────────────────────────────
python scripts/evaluate.py --model_path models/checkpoints --split test

# ── Inference ───────────────────────────────────
python scripts/transcribe.py --model_path models/checkpoints --audio audio.wav

# ── Monitoring ──────────────────────────────────
tensorboard --logdir models/checkpoints/logs
#   → Open http://localhost:6006
```
