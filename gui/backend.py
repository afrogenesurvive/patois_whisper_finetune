"""
GUI Backend — thin orchestration layer that imports existing ``scripts.*``
functions and wraps them for Gradio integration.

No changes to the CLI scripts are needed — the GUI calls them as a library.
"""

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gui.state import training_state
from gui.utils import read_tb_scalar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data tab — wraps scripts.prepare_data
# ---------------------------------------------------------------------------

_processed_paths: List[Path] = []


def run_resample(raw_dir: str = "data/raw") -> Tuple[bool, str]:
    """Resample all audio in *raw_dir* to 16 kHz mono WAV."""
    try:
        from scripts.prepare_data import resample_audio

        global _processed_paths
        _processed_paths = resample_audio(Path(raw_dir), Path("data/processed"))
        return (
            True,
            f"Resampled {len(_processed_paths)} files to data/processed/",
        )
    except Exception as e:
        logger.exception("Resample failed")
        return False, f"Error: {e}"


def run_pseudo_label(
    model_size: str = "medium",
    batch_size: int = 8,
) -> Tuple[bool, str, Dict[str, str]]:
    """Generate pseudo-labels for processed audio files."""
    try:
        from scripts.prepare_data import generate_pseudo_labels

        global _processed_paths
        if not _processed_paths:
            _processed_paths = sorted(Path("data/processed").glob("*.wav"))

        transcripts = generate_pseudo_labels(
            _processed_paths, model_size=model_size, batch_size=batch_size
        )
        return True, f"Generated {len(transcripts)} pseudo-labels", transcripts
    except Exception as e:
        logger.exception("Pseudo-label generation failed")
        return False, f"Error: {e}", {}


def save_correction(stem: str, text: str, transcript_dir: str = "data/transcripts") -> str:
    """
    Save a single corrected transcript to *transcript_dir*.

    Appends to a cumulative JSON file (``all_corrections.json``) so the
    user can correct incrementally without managing multiple files.
    """
    transcript_dir = Path(transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    json_path = transcript_dir / "all_corrections.json"
    if json_path.exists():
        with open(json_path) as f:
            corrections = json.load(f)
    else:
        corrections = {}

    corrections[stem] = text
    with open(json_path, "w") as f:
        json.dump(corrections, f, indent=2, ensure_ascii=False)

    return f"Saved correction for '{stem}'"


def get_corrections(transcript_dir: str = "data/transcripts") -> Dict[str, str]:
    """Load all saved corrections from *transcript_dir*."""
    from scripts.prepare_data import load_corrected_transcripts
    return load_corrected_transcripts(Path(transcript_dir))


def run_build_dataset(
    val_split: float = 0.1,
    test_split: float = 0.1,
    dataset_dir: str = "data/dataset",
) -> Tuple[bool, str, dict]:
    """
    Build the Hugging Face DatasetDict from corrected transcripts and
    resampled audio.
    """
    try:
        from scripts.prepare_data import create_dataset_dict, save_dataset_info

        corrected = get_corrections()
        if not corrected:
            return False, "No corrected transcripts found — correct some first!", {}

        dataset_dict = create_dataset_dict(
            Path("data/processed"),
            corrected,
            val_split=val_split,
            test_split=test_split,
        )

        # Save to disk
        dataset_dict.save_to_disk(str(dataset_dir))
        save_dataset_info(dataset_dict, Path(dataset_dir))

        sizes = {
            "train": len(dataset_dict["train"]),
            "validation": len(dataset_dict["validation"]),
            "test": len(dataset_dict["test"]),
        }
        return True, "Dataset built successfully!", sizes

    except Exception as e:
        logger.exception("Build dataset failed")
        return False, f"Error: {e}", {}


# ---------------------------------------------------------------------------
# Train tab — runs training in a background thread
# ---------------------------------------------------------------------------

_train_thread: Optional[threading.Thread] = None


def _train_worker(config: dict) -> None:
    """Run training in a background thread, updating *training_state*."""
    import io

    # Redirect stdout to capture training logs
    log_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = log_capture

    training_state.update(running=True, start_time=__import__("time").time())

    try:
        from scripts.train import (
            load_config,
            load_model,
            load_processor,
            load_datasets,
            compute_wer,
            get_trainer,
            train_and_save,
        )
        from transformers import TrainerCallback

        # ── Custom callback for stop support ────────────────────────
        class GUIControlCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if training_state.stop_requested:
                    control.should_training_stop = True
                    training_state.add_log("⏹ Stop requested — finishing step...")
                # Update state from trainer
                training_state.update(
                    current_step=state.global_step,
                    total_steps=state.max_steps,
                )
                # Snapshot latest loss from log history
                if state.log_history:
                    last = state.log_history[-1]
                    if "loss" in last:
                        training_state.update(current_loss=last["loss"])
                    if "eval_wer" in last:
                        training_state.update(current_wer=last["eval_wer"])
                    if "learning_rate" in last:
                        training_state.update(current_lr=last["learning_rate"])
                return control

        # ── Build & run trainer ─────────────────────────────────────
        model_name = config["model"]["name_or_path"]
        use_lora = config["lora"]["enabled"]

        processor, _fe, tokenizer = load_processor(model_name)
        model = load_model(model_name, use_lora=use_lora, lora_config=config.get("lora"))

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        dataset_dict, _test_ds = load_datasets(config, processor)
        train_ds = dataset_dict[config["data"]["train_split"]]
        val_ds = dataset_dict[config["data"]["val_split"]]

        def metrics_fn(preds, labels):
            return compute_wer(preds, labels, tokenizer)

        trainer = get_trainer(
            config, model, tokenizer, train_ds, val_ds, metrics_fn,
        )

        # Add the stop callback
        trainer.add_callback(GUIControlCallback)

        train_and_save(trainer, config, use_lora=use_lora)

        # ── Final eval ──────────────────────────────────────────────
        training_state.add_log("✅ Training complete!")

    except Exception as e:
        logger.exception("Training failed")
        training_state.add_log(f"❌ Training failed: {e}")
    finally:
        sys.stdout = old_stdout
        training_state.update(running=False)


def start_training(config_path: str = "config.yaml") -> str:
    """Start training in a background thread."""
    global _train_thread

    if training_state.running:
        return "Training is already running!"

    # Reload config from file
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    training_state.reset_stop()
    training_state.update(
        running=False,
        current_step=0,
        current_loss=None,
        current_wer=None,
        logs=[],
    )

    _train_thread = threading.Thread(
        target=_train_worker,
        args=(config,),
        daemon=True,
    )
    _train_thread.start()
    return "Training started in background."


def stop_training() -> str:
    """Request graceful stop of training."""
    if not training_state.running:
        return "No training is running."
    training_state.request_stop()
    return "Stop requested. Waiting for current step to finish..."


# ---------------------------------------------------------------------------
# Evaluate tab — wraps scripts.evaluate
# ---------------------------------------------------------------------------

def run_evaluation(
    model_path: str,
    dataset_path: str = "data/dataset",
    split: str = "test",
    batch_size: int = 8,
) -> Tuple[bool, dict]:
    """Run evaluation on a trained model."""
    try:
        from scripts.evaluate import load_model_and_processor, evaluate
        from datasets import load_from_disk
        import torch

        model, processor = load_model_and_processor(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

        dataset_dict = load_from_disk(dataset_path)
        test_dataset = dataset_dict[split]

        results = evaluate(
            test_dataset, model, processor, device,
            batch_size=batch_size,
            output_path=f"{model_path}/evaluation_results.json",
        )
        return True, results
    except Exception as e:
        logger.exception("Evaluation failed")
        return False, {"error": str(e)}


# ---------------------------------------------------------------------------
# Infer tab — wraps scripts.transcribe
# ---------------------------------------------------------------------------

def run_transcription(
    audio_path: str,
    model_path: str,
) -> Tuple[bool, str]:
    """Transcribe a single audio file."""
    try:
        from scripts.transcribe import load_model_and_processor, transcribe_file
        import torch

        model, processor = load_model_and_processor(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

        text = transcribe_file(audio_path, model, processor, device)
        return True, text
    except Exception as e:
        logger.exception("Transcription failed")
        return False, f"Error: {e}"
