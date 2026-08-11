# Patois Whisper Fine-Tune

Fine-tune a Whisper model to transcribe **Jamaican Patois audio** into **plain English text**.

**Approach**: Self-training with pseudo-labeling — generate initial transcripts with a base Whisper model, manually correct them to accurate English translations, then fine-tune on the corrected pairs.

**Proven results** (from research):

- 40+ hours of corrected audio → **WER 0.30** with `whisper-medium` (down from 0.89 untrained)
- Even 20–35 hours yields significant gains

---

## Project Structure

```
patois_whisper_finetune/
├── data/
│   ├── raw/              # Original audio files (any format)
│   ├── processed/        # Resampled 16kHz mono WAV/FLAC
│   ├── transcripts/      # Corrected transcripts (JSON/CSV)
│   └── dataset/          # Hugging Face Dataset artifacts
├── scripts/
│   ├── prepare_data.py   # Audio resampling + dataset formatting
│   ├── train.py          # Fine-tuning (standard + LoRA)
│   ├── evaluate.py       # WER evaluation on a test set
│   └── transcribe.py     # Inference with the fine-tuned model
├── models/               # Checkpoints and LoRA adapters (gitignored)
├── config.yaml           # Hyperparameters
├── requirements.txt      # Python dependencies
└── setup_environment.sh  # Environment setup script
```

---

## End-to-End Workflow

### Step 1: Prepare Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or use the automated script:

```bash
bash setup_environment.sh
```

### Step 2: Collect and Organize Audio

Place raw audio files in `data/raw/`. Create `data/transcripts/` directory if needed.

### Step 3: Generate Pseudo-Labels & Correct

Generate initial Whisper transcripts:

```bash
python scripts/prepare_data.py --raw_dir data/raw --model_size medium
```

This resamples audio to 16kHz mono, runs Whisper inference, and prints the auto-generated transcripts. **Manually correct** these transcripts to accurate English and save them in `data/transcripts/` (as JSON or CSV).

Then re-run to build the dataset:

```bash
python scripts/prepare_data.py --skip_pseudo
```

### Step 4: Train

Standard fine-tune:

```bash
python scripts/train.py --config config.yaml
```

Or with LoRA (faster, smaller output):

```bash
python scripts/train.py --config config.yaml --use_lora
```

Resume from a checkpoint:

```bash
python scripts/train.py --config config.yaml --resume models/checkpoints/checkpoint-1000
```

### Step 5: Evaluate

```bash
python scripts/evaluate.py --model_path models/checkpoints --split test
```

### Step 6: Transcribe New Audio

```bash
python scripts/transcribe.py --model_path models/checkpoints --audio new_audio.wav
```

### Step 7: The Data Flywheel

After deploying, use the improved model to transcribe **new**, unlabeled audio. Lightly correct those transcripts and add them to the training set. Retrain periodically to continuously improve.

---

## Reference Benchmarks

| Configuration                      | WER   |
| ---------------------------------- | ----- |
| Untrained Whisper-medium on Patois | ~0.89 |
| After 40h fine-tune (medium)       | ~0.30 |
| After 40h fine-tune (small)        | ~0.51 |

- **Target dataset size**: 40+ hours of corrected audio
- **Training steps**: 4,000 total
- **Learning rate**: 1e-5 with 500-step linear warmup
- **Effective batch size**: 16 (8 per device × 2 gradient accumulation)

---

## Delivery Options

### Standard: Full Model Weights

Replace the off-the-shelf Whisper model in your pyannote+Whisper pipeline with the fine-tuned one from `models/checkpoints/`. No code changes needed.

### Efficient: LoRA Adapter

Deliver only the adapter weights (a few MB). The client loads the base Whisper model and applies the LoRA adapter on top. Use `--use_lora` during training to produce adapter-only checkpoints. The adapter files (`adapter_config.json`, `adapter_model.safetensors`) are in `models/checkpoints/`.

---

## GUI — Web Interface

A Gradio-based web GUI is available for users who prefer a visual interface over the command line:

```bash
# Install GUI dependencies
pip install gradio plotly

# Launch the GUI
python gui/app.py
# → Opens at http://127.0.0.1:7860

# Optional: share with others via a public link
python gui/app.py --share
```

---

## macOS App — Native Desktop Application

A native macOS `.app` bundle is available. It wraps the Gradio web interface in a
`pywebview` window so you can launch from Finder / Dock — no terminal needed.

### Prerequisites

```bash
pip install pywebview py2app
```

Or use the Makefile:

```bash
make deps
```

### Quick Dev Loop (no .app build)

Test the native window without building the full bundle:

```bash
make run
```

This runs `python gui/launcher.py` directly — the same entry point the `.app` uses.

### Build the .app Bundle

```bash
make app
```

After the build completes, the app is at:

```
dist/Whisper Fine-Tune GUI.app
```

Drag it to `/Applications/` and launch it like any other Mac app.

### First Launch

The `.app` is **lightweight (~200–300 MB)** — PyTorch is excluded from the bundle
to save 2–4 GB. On first launch, the app detects whether PyTorch is installed:

- **Installed** → The Gradio GUI opens immediately in a native window.
- **Not installed** → A setup screen appears with an **Install PyTorch** button.
  Clicking it runs `pip install torch` (CPU-only, appropriate for macOS) and then
  automatically loads the GUI.

> **Note**: The `gradio` package analytics are disabled by default in the launcher
> (`GRADIO_ANALYTICS_ENABLED=False`).

### Logging & Troubleshooting

The launcher writes a diagnostic log to:

```
~/.whisper-gui/app.log
```

Each launch starts a fresh log; the previous run is kept at `~/.whisper-gui/app.log.prev`.
The log captures startup errors, Gradio server status, and uncaught exceptions (raw
`stderr`/`stdout` is redirected into it when the app is launched from Finder/Dock).
Check it first when the app won't start or the window stays on the setup screen.

### Custom Icon

To add a custom app icon, place a 1024×1024 `icon.icns` file at `gui/icon.icns`.
No code changes needed — py2app will pick it up if you add `"iconfile": "gui/icon.icns"`
to the `OPTIONS` dict in `setup_app.py`.
```

The GUI has four tabs:

| Tab             | Purpose                                                                         |
| --------------- | ------------------------------------------------------------------------------- |
| **🗂 Data**     | Upload audio, generate pseudo-labels, correct transcripts inline, build dataset |
| **🎓 Train**    | Configure hyperparameters, start/stop training, view live loss/WER charts       |
| **📊 Evaluate** | Run WER/CER evaluation, browse error analysis table                             |
| **🎙 Infer**    | Transcribe audio files (single or batch) with a trained model                   |

**No changes to existing CLI scripts** — the GUI imports them as a Python library. You can continue using the terminal workflow side-by-side.

See `docs/training_manual.md` for a complete walkthrough.

---

## Configuration

See `config.yaml` for all training hyperparameters. Key settings:

- `model.name_or_path`: Whisper model variant (`openai/whisper-medium` or `openai/whisper-small`)
- `training.num_steps`: Total training steps (default: 4000)
- `lora.enabled`: Toggle LoRA fine-tuning (default: false)
- `data.dataset_path`: Path to the prepared dataset

---

## Requirements

- Python 3.8+
- CUDA-enabled GPU with 8+ GB VRAM (recommended)
- See `requirements.txt` for Python package dependencies
