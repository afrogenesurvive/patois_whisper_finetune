# Changelog

## [0.3.0] - 2026-07-17

### Added

- **`gui/`** — new Gradio-based web interface for the Whisper fine-tuning pipeline
- **`gui/app.py`** — main entry point with 4-tab layout, CLI args for port/share/debug
- **`gui/backend.py`** — orchestration layer importing `scripts.*` as a library; wraps data prep, training (background thread with stop support), evaluation, and inference
- **`gui/state.py`** — thread-safe `TrainingState` dataclass with stop event, log buffer, and snapshot helpers
- **`gui/utils.py`** — TB event reader (`EventAccumulator`), checkpoint lister, config loader/saver, time formatter
- **`gui/tabs/data_tab.py`** — Data tab: audio upload/record, resample, pseudo-label generation, inline transcript correction (click row → edit → save), dataset builder
- **`gui/tabs/train_tab.py`** — Train tab: config form (model, mode, hyperparams), save config, start/stop training, live Plotly charts (loss/WER via TB polling), console log with periodic refresh
- **`gui/tabs/evaluate_tab.py`** — Evaluate tab: checkpoint selector, WER/CER display, sortable error analysis table (reference vs hypothesis), JSON export
- **`gui/tabs/infer_tab.py`** — Infer tab: single-file transcription (upload or record), batch transcription with ZIP download
- **`gui/project_config.yaml`** — extensible project configuration for future multi-task support
- `requirements.txt` — added `gradio>=4.0.0`, `plotly>=5.15.0`
- `README.md` — added GUI section with launch instructions and tab overview
- `.gitignore` — added `gui/__pycache__/`

## [0.2.0-3] - 2026-07-17

### Changed

- `docs/training_manual.md` — auto-formatted markdown tables and spacing for consistent readability

## [0.2.0-2] - 2026-07-17

### Added

- **`docs/training_manual.md`** — comprehensive training instruction manual covering setup, installation, data preparation, training, evaluation, inference, delivery, the data flywheel, troubleshooting, and FAQ

## [0.2.0-1] - 2026-07-17

### Added

- `setup_environment.sh` made executable
- Verification: all 4 Python scripts pass AST parsing, config.yaml parses correctly, and shell script syntax is valid
- Full CLI help verified for all `scripts/*.py` scripts

## [0.2.0] - 2026-07-17

### Added

- **`scripts/prepare_data.py`** — data preparation pipeline: audio resampling to 16kHz mono, pseudo-label generation with base Whisper, corrected transcript loading (JSON/CSV), and Hugging Face DatasetDict creation with train/val/test splits
- **`scripts/train.py`** — training script supporting standard fine-tune, LoRA (`--use_lora`), and resume (`--resume`). Uses `Seq2SeqTrainer` with `predict_with_generate=True`, `DataCollatorSpeechSeq2SeqWithPadding`, and TensorBoard logging
- **`scripts/evaluate.py`** — evaluation script with batched inference, WER/CER computation, and detailed error analysis table (reference vs hypothesis pairs)
- **`scripts/transcribe.py`** — single-file inference script with auto LoRA detection, supports any librosa-compatible audio format
- **`scripts/__init__.py`** — package init for clean imports
- **`config.yaml`** — YAML hyperparameter config (Whisper-medium, 4k steps, 1e-5 LR, 500 warmup, LoRA section, data section)
- **`requirements.txt`** — pinned dependencies (torch, transformers, datasets, accelerate, evaluate, jiwer, librosa, soundfile, peft, pyyaml, tqdm, tensorboard)
- **`setup_environment.sh`** — automated environment setup: creates venv, detects CUDA, installs PyTorch + requirements
- **`README.md`** — full project documentation with end-to-end workflow, reference benchmarks, and delivery options
- `data/raw/`, `data/processed/`, `data/transcripts/`, `data/dataset/`, `models/` — project directory structure with `.gitkeep` files

## [0.1.0] - 2026-07-17

### Added

- `agent_scaffold_instructions.md` — distilled instructions for scaffolding the Whisper Patois fine-tuning project, covering project structure, environment setup, data preparation, training (standard + LoRA), evaluation, and inference
- `docs/` directory with project documentation
- `.gitignore` with exclusions for `safe/`, data artifacts, model checkpoints, and standard Python/OS files
