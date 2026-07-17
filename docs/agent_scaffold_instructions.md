# Agent Scaffold Instructions: Whisper Fine-Tune for Jamaican Patois

## Project Overview

Fine-tune a Whisper model (small or medium) to transcribe Jamaican Patois audio into **plain English text**. The approach uses **self-training with pseudo-labeling**: generate initial transcripts with a base Whisper model, manually correct them to accurate English translations, then fine-tune on the corrected pairs.

**Key benchmarks from research:**

- 40+ hours of corrected audio yields WER **0.30** with `whisper-medium` (down from 0.89 untrained)
- Even 20–35 hours produces significant gains
- Smaller models (`small`) also benefit substantially

---

## 1. Project Structure to Scaffold

Create the following directory layout:

```
patois_whisper_finetune/
├── data/
│   ├── raw/              # Original audio files (any format)
│   ├── processed/        # Resampled 16kHz mono WAV/FLAC
│   ├── transcripts/      # Corrected transcripts (JSON/CSV)
│   └── dataset/          # Hugging Face Dataset artifacts (optional cache)
├── scripts/
│   ├── prepare_data.py   # Audio resampling + dataset formatting
│   ├── train.py          # Fine-tuning (standard + LoRA with --use_lora)
│   ├── evaluate.py       # WER evaluation on a test set
│   └── transcribe.py     # Inference with the fine-tuned model
├── models/               # Checkpoints and LoRA adapters (gitignored)
├── requirements.txt
└── config.yaml           # Hyperparameters
```

Also create a `.gitignore` that excludes `data/processed/`, `data/dataset/`, `models/`, `__pycache__/`, `*.egg-info/`.

---

## 2. Environment Setup

Create `requirements.txt` with the following dependencies. Pin major versions but allow patch flexibility:

- `torch>=2.0.0` (with CUDA)
- `transformers>=4.30.0`
- `datasets>=2.14.0`
- `accelerate>=0.20.0`
- `evaluate`
- `jiwer` (for WER computation)
- `librosa` (audio loading)
- `soundfile` (audio I/O)
- `peft>=0.5.0` (for LoRA support)
- `tensorboard` or `wandb` (optional, for logging)
- `pyyaml` (for config loading)
- `tqdm`

The agent should also create a `setup_environment.sh` (or document instructions) that:

1. Creates a Python virtual environment (or conda env)
2. Installs PyTorch with the appropriate CUDA version
3. Installs the remaining requirements

---

## 3. Data Preparation Script (`scripts/prepare_data.py`)

The agent should create a script with the following structure and logic:

### CLI Interface

```python
# Usage:
#   python scripts/prepare_data.py \
#       --raw_dir data/raw \
#       --output_dir data/processed \
#       --transcript_dir data/transcripts \
#       --model_size small \
#       --val_split 0.1 \
#       --test_split 0.1
```

### Functions to implement

| Function                                                             | Signature                                     | Description                                                                                                                                                                                                                                                                    |
| -------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `resample_audio(raw_dir, output_dir, target_sr=16000)`               | `(Path, Path, int) -> List[Path]`             | Walk `raw_dir`, load each audio file with librosa, resample to `target_sr` Hz mono, save as 16-bit WAV to `output_dir`. Return list of processed file paths.                                                                                                                   |
| `generate_pseudo_labels(audio_paths, model_size, batch_size)`        | `(List[Path], str, int) -> Dict[str, str]`    | Load a base Whisper model+processor from Hugging Face using `model_size` (`"small"` or `"medium"`). Transcribe each audio file. Return a dict mapping filename stem → raw transcript text.                                                                                     |
| `load_corrected_transcripts(transcript_dir)`                         | `(Path) -> Dict[str, str]`                    | Load manually corrected transcript files from `transcript_dir`. Support JSON format: `{"filename": "transcript text", ...}` and CSV format with columns `filename,transcript`. Return a dict.                                                                                  |
| `create_dataset_dict(audio_dir, transcripts, val_split, test_split)` | `(Path, Dict, float, float) -> DatasetDict`   | Build a Hugging Face `DatasetDict` with `cast_column("audio", Audio(sampling_rate=16000))`. The dataset should have columns: `audio` (path), `text` (transcript). Split using `val_split` and `test_split` fractions. Save a preprocessed copy to `data/dataset/` for caching. |
| `prepare_dataset(audio_features, tokenizer, max_length)`             | `(Dataset, WhisperTokenizer, int) -> Dataset` | Apply the Whisper feature extractor to the audio column and tokenizer to the text column. Return a transformed dataset ready for training.                                                                                                                                     |

### Processing pipeline (main block)

1. Parse CLI args using `argparse`
2. Call `resample_audio()` to normalize all audio
3. Call `generate_pseudo_labels()` to get base Whisper transcripts
4. Print pseudo-labels and prompt user: _"Correct these transcripts in data/transcripts/, then re-run with --skip_pseudo"_
5. On re-run (or if `--skip_pseudo`): call `load_corrected_transcripts()`
6. Merge audio paths with corrected transcripts
7. Call `create_dataset_dict()` to produce train/val/test splits
8. Save a `dataset_info.json` with split sizes, sample rates, and a few example rows for inspection

---

## 4. Config File (`config.yaml`)

Create a YAML configuration file with the following parameters (derived from proven Patois research):

```yaml
model:
  name_or_path: "openai/whisper-small" # or "openai/whisper-medium"
  language: "en"
  task: "transcribe"
  sampling_rate: 16000

training:
  output_dir: "models/checkpoints"
  num_steps: 4000
  warmup_steps: 500
  learning_rate: 1.0e-5
  optimizer: "adamw_torch"
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  per_device_eval_batch_size: 8
  fp16: true
  logging_steps: 50
  eval_steps: 200
  save_steps: 500
  save_total_limit: 3
  metric_for_best_model: "wer"
  greater_is_better: false
  load_best_model_at_end: true

lora:
  enabled: false
  r: 8
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]
  lora_dropout: 0.05
  bias: "none"

data:
  dataset_path: "data/dataset"
  train_split: "train"
  val_split: "validation"
  test_split: "test"
  max_audio_length: 30.0 # seconds, trim longer clips
```

---

## 5. Training Script (`scripts/train.py`)

The agent should create a training script with these components:

### CLI Interface

```python
# Usage:
#   python scripts/train.py --config config.yaml                    # Standard fine-tune
#   python scripts/train.py --config config.yaml --use_lora          # LoRA fine-tune
#   python scripts/train.py --config config.yaml --resume checkpoint-1000  # Resume
```

### Functions to implement

| Function                                                                              | Signature                                                                | Description                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_config(config_path)`                                                            | `(str) -> dict`                                                          | Load YAML config, merge with CLI overrides. Return config dict.                                                                                                                                                                    |
| `load_model(model_name, use_lora, lora_config)`                                       | `(str, bool, dict) -> (WhisperForConditionalGeneration, PeftModel)`      | Load the Whisper model from Hugging Face. If `use_lora`, wrap with `peft.LoraModel` using the LoRA config (r, alpha, target modules, dropout). Freeze the base model params. Return model (and optionally the PEFT model wrapper). |
| `load_processor(model_name)`                                                          | `(str) -> (WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer)` | Load the Whisper processor, feature extractor, and tokenizer.                                                                                                                                                                      |
| `load_datasets(config)`                                                               | `(dict) -> (DatasetDict, Dataset)`                                       | Load the prepared dataset from `config["data"]["dataset_path"]`. Apply the feature extraction transform using `map()` with the processor. Return the split dataset dict and the test set.                                          |
| `compute_wer(preds, labels, tokenizer)`                                               | `(np.array, np.array, WhisperTokenizer) -> dict`                         | Decode predictions and labels, compute WER via `jiwer`, return `{"wer": float}`.                                                                                                                                                   |
| `get_trainer(config, model, tokenizer, train_dataset, eval_dataset, compute_metrics)` | `(dict, model, tokenizer, Dataset, Dataset, callable) -> Trainer`        | Configure Hugging Face `Seq2SeqTrainer` with: training args from config (num_steps, warmup, LR, fp16, eval/save strategies), data collator for Whisper, and the WER compute function.                                              |
| `train_and_save(trainer, config)`                                                     | `(Trainer, dict) -> None`                                                | Call `trainer.train()`, then save the final model and tokenizer to `config["training"]["output_dir"]`. If LoRA, save only the adapter weights.                                                                                     |

### Main block flow

1. Parse `--config`, `--use_lora` (flag), `--resume` (optional checkpoint path)
2. Load config YAML
3. Load processor, model (optionally with LoRA)
4. Load and prepare datasets
5. Initialize `Seq2SeqTrainer`
6. Train
7. Save final artifacts

### Important details to include

- The `Seq2SeqTrainingArguments` should set `predict_with_generate=True`
- Generation config: `num_beams=5`, `max_length=128`
- Use `DataCollatorSpeechSeq2SeqWithPadding` from the Transformers library
- Log WER to TensorBoard/WandB
- On resume, load optimizer/scheduler state too

---

## 6. Evaluation Script (`scripts/evaluate.py`)

### CLI Interface

```bash
# Usage:
#   python scripts/evaluate.py --model_path models/checkpoints \
#       --dataset_path data/dataset --split test
```

### Functions

| Function                                                              | Signature                                              | Description                                                                                                                                                                  |
| --------------------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_model_and_processor(model_path, use_lora)`                      | `(str, bool) -> (model, processor)`                    | Load the fine-tuned model. If a LoRA adapter exists (look for `adapter_config.json` in `model_path`), load base model first then apply PEFT adapter.                         |
| `transcribe_batch(audio_paths, model, processor, device, batch_size)` | `(List[str], model, processor, str, int) -> List[str]` | Run inference in batches. Generate token IDs, decode to text.                                                                                                                |
| `evaluate(test_dataset, model, processor, device)`                    | `(Dataset, model, processor, str) -> dict`             | Transcribe all test samples, compute WER against ground truth. Return dict with `wer`, `cer` (optional), and a table of (reference, hypothesis) pairs for manual inspection. |

Print results to stdout and optionally save the error analysis table to `models/evaluation_results.json`.

---

## 7. Inference Script (`scripts/transcribe.py`)

### CLI Interface

```bash
# Usage:
#   python scripts/transcribe.py --model_path models/checkpoints \
#       --audio path/to/audio.wav [--output text]
```

### Functions

| Function                                                | Signature                             | Description                                                                                                     |
| ------------------------------------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `transcribe_file(audio_path, model, processor, device)` | `(str, model, processor, str) -> str` | Load a single audio file, resample to 16kHz mono if needed, run Whisper inference, return the transcribed text. |
| `main()`                                                | `() -> None`                          | Parse args, load model (with LoRA detection same as evaluate.py), transcribe, print or save output.             |

Include a `--device` flag (default: auto-detect CUDA vs CPU).

---

## 8. Usage Workflow (to document in the file)

The distilled document should include this end-to-end workflow section for the human operator:

### Step 1: Prepare Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Collect and Organize Audio

Place raw audio files in `data/raw/`. Create `data/transcripts/` directory.

### Step 3: Generate Pseudo-Labels & Correct

```bash
python scripts/prepare_data.py --raw_dir data/raw --model_size small
```

This generates initial transcripts. Manually correct them in `data/transcripts/`. Then re-run:

```bash
python scripts/prepare_data.py --skip_pseudo
```

### Step 4: Train

```bash
# Standard fine-tune
python scripts/train.py --config config.yaml

# Or with LoRA (faster, smaller output)
python scripts/train.py --config config.yaml --use_lora
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

After deploying, use the improved model to transcribe _new_ unlabeled audio. Lightly correct those transcripts and add them to the training set. Retrain periodically to continuously improve.

---

## 9. Reference Benchmarks (to include)

- **Untrained Whisper-medium on Patois**: WER ~0.89
- **After 40h fine-tune (medium)**: WER ~0.30
- **Target dataset size**: 40+ hours of corrected audio
- **Training steps**: 4,000 total
- **Learning rate**: 1e-5 with 500-step linear warmup
- **Effective batch size**: 16 (8 per device × 2 gradient accumulation)

---

## 10. Delivery Options (to document)

### Standard: Full Model Weights

The client replaces the off-the-shelf Whisper model in their pyannote+Whisper pipeline with the fine-tuned one from `models/checkpoints/`. No code changes needed.

### Efficient: LoRA Adapter

Deliver only the adapter weights (a few MB). The client loads the base Whisper model and applies the LoRA adapter on top. Use `--use_lora` during training to produce adapter-only checkpoints. The adapter files (`adapter_config.json`, `adapter_model.safetensors`) are in `models/checkpoints/`.
