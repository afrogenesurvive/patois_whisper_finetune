# Changelog

## [0.0.1-2] - 2026-08-21

### Added

- **Floating "✕ Quit App" button** — added to both the Gradio GUI and the launcher
  setup screen, with a confirmation dialog, so you can quit the app from inside the
  window instead of `Cmd+Q`.
- **`setup_app.py`** — the macOS `.app` now **bundles PyTorch (CPU)** at build time,
  so it runs out-of-the-box (the bundle is larger, ~1 GB). The runtime "Install
  PyTorch" setup step is no longer needed.

### Changed

- Text in the app window is now selectable/copyable.

## [0.0.1-1] - 2026-08-11

### Added

- **`gui/launcher.py`** — `_setup_file_logging()`: file logging is now configured at import time (not just inside `main()`), writing to `~/.whisper-gui/app.log` and rotating the previous run to `app.log.prev`
- **`gui/launcher.py`** — when launched from Finder/Dock (no TTY), `sys.stderr`/`sys.stdout` are redirected into the log file so uncaught tracebacks and `print()` output are captured for diagnosis
- **`gui/launcher.py`** — `_pick_port()` and a retry loop in `_start_gradio_server()`: if the chosen port is occupied (e.g. grabbed during the long Gradio import), the app re-picks a free port and retries up to 5 times instead of failing
- **`gui/launcher.py`** — `_Api.get_gradio_url()` JS bridge; the setup page now fetches the live Gradio URL before navigating, so it follows a port retry
- **`setup_app.py`** — bundle the uvicorn/anyio modules loaded via dynamic string imports (`uvicorn.loops.*`, `uvicorn.protocols.http.*`, `uvicorn.protocols.websockets.*`, `uvicorn.lifespan.*`, `uvicorn.__main__`, `anyio._backends.*`) so a clean py2app build can start the Gradio server
- `.gitignore` — ignore the local `.github/prompts/` directory

### Fixed

- **`gui/launcher.py`** — log file is opened with UTF-8 encoding; previously em-dashes in log messages crashed the handler with `UnicodeEncodeError: 'ascii' codec can't encode '\u2014'` under the `.app`'s ASCII-default Python
- **`.app` startup** — the recurring "Cannot find empty port in range: 7860-7860" was a misleading symptom of missing dynamically-imported uvicorn/anyio modules; fixed by bundling them (see Added)
- **`gui/launcher.py`** — on a busy port, the log now records which process holds it (via `lsof`) before retrying, aiding diagnosis

## [0.3.0-3] - 2026-07-24

### Added

- **`gui/launcher.py`** — `_wait_for_server()` TCP readiness check to prevent JS from navigating to Gradio URL before server is ready (eliminates "connection errored" notifications in pywebview)
- **`gui/launcher.py`** — `_find_python_for_pip()` with 4-step fallback for py2app bundles where `sys.executable` points to a non-runnable stub
- **`gui/launcher.py`** — `_Api.navigate_to()` using `_window.load_url()` to fix navigation from setup page to Gradio UI (pywebview windows created with `html=` don't support `window.location.href`)
- **`gui/launcher.py`** — `_Api.get_recent_logs()` JS bridge for real-time log polling on the setup page
- **`gui/launcher.py`** — Setup page collapsible "Console Log" dark-terminal panel with 1-second polling interval
- **`gui/launcher.py`** — File logging to `~/.whisper-gui/app.log` via `logging.FileHandler` in `main()`
- **`gui/utils.py`** — `LogBufferHandler` class (deque-backed `logging.Handler`) and `log_buffer` singleton for shared in-app log viewing across pywebview and Gradio
- **`gui/app.py`** — "📋 App Logs" accordion in Gradio UI with `gr.Timer(3)` periodic refresh from the shared log buffer

### Fixed

- **`gui/launcher.py`** — `_start_gradio_server()` now creates an asyncio event loop before `build_app()` to prevent `RuntimeError: There is no current event loop in thread` when Gradio's `safe_get_lock()` constructs `asyncio.Lock()` on a background thread (Python 3.9 compat)
- **`setup_app.py`** — Removed `matplotlib` from `OPTIONS["excludes"]` to prevent build warnings

## [0.3.0-2] - 2026-07-21

### Added

- **`Makefile`** — build automation for `app`, `run`, `clean`, and `deps` targets
- **`gui/launcher.py`** — native macOS app launcher wrapping Gradio in a `pywebview` window; dynamic dependency checking (detects missing PyTorch and offers in-app install)
- **`setup_app.py`** — `py2app` build script for creating `dist/Whisper Fine-Tune GUI.app`
- `README.md` — added macOS native app section with prerequisites, build instructions, and first-launch notes
- `requirements.txt` — added `pywebview>=4.0` and `py2app>=0.28`

### Fixed

- **macOS `.app` launch crash** — added missing `from typing import Optional` import in `gui/launcher.py` (caused `NameError` at startup)
- **`gui/utils.py`** — deferred `EventAccumulator` import to inside functions with `ImportError` guards, so TensorBoard import failures don't break the GUI
- **`setup_app.py` PyObjC packages** — added `objc`, `AppKit`, `Foundation`, `WebKit`, `CoreFoundation`, `Quartz` to the `packages` list; required by `webview.platforms.cocoa` for the native macOS window
- **Build process** — `Makefile` targets now use `./venv/bin/python3` instead of bare `python3` to avoid picking up the system Python which lacks `modulegraph`

## [0.3.0-1] - 2026-07-19

### Added

- **`docs/training_manual.md`** — comprehensive GUI documentation:
  - Added §3.4 Launching the GUI with CLI flags reference table
  - Added §3.5 The GUI Interface with tab overview table
  - Added GUI project structure listing in §2
  - Added GUI workflow tip in §4 Data Preparation
  - Added GUI tab references in §5 Training, §6 Evaluation, §7 Inference
  - Added §11.6 GUI troubleshooting section (port conflicts, imports, blank page, checkpoints)
  - Added GUI check item to debugging checklist
- `gui/tabs/data_tab.py` — improved file table display and correction workflow
- `gui/tabs/train_tab.py` — enhanced config form and chart rendering
- `gui/app.py` — refined UI layout and error handling

### Fixed

- `venv` (local) — hot-patched `gradio_client/utils.py` to handle boolean `additionalProperties` in JSON schema, fixing `TypeError: argument of type 'bool' is not iterable` on startup

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
