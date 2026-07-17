"""
Training script for Whisper fine-tuning on Jamaican Patois.

Supports:
  - Standard full fine-tune
  - LoRA fine-tune (parameter-efficient)
  - Resume from checkpoint

Usage:
  python scripts/train.py --config config.yaml                    # Standard
  python scripts/train.py --config config.yaml --use_lora          # LoRA
  python scripts/train.py --config config.yaml --resume checkpoint-1000  # Resume
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import evaluate as hf_evaluate
import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    DataCollatorSpeechSeq2SeqWithPadding,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config and return as dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


# ---------------------------------------------------------------------------
# 2. Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    use_lora: bool = False,
    lora_config: Optional[dict] = None,
    resume_checkpoint: Optional[str] = None,
) -> WhisperForConditionalGeneration:
    """
    Load the Whisper model from Hugging Face.

    If ``use_lora``, wrap with PEFT LoRA using the provided config and
    freeze base model parameters.  Returns the model (and PEFT wrapper
    if applicable).
    """
    logger.info(f"Loading model: {model_name}")

    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        model = WhisperForConditionalGeneration.from_pretrained(resume_checkpoint)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Freeze encoder for LoRA (optional best practice for whisper)
    if use_lora:
        if lora_config is None:
            raise ValueError("lora_config required when use_lora=True")

        logger.info("Applying LoRA adapters")
        peft_config = LoraConfig(
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("lora_alpha", 32),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Set generation config
    model.config.forced_decoder_ids = None  # Let tokenizer handle language
    model.config.suppress_tokens = []

    return model


def load_processor(
    model_name: str,
) -> Tuple[WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer]:
    """Load the Whisper processor, feature extractor, and tokenizer."""
    logger.info(f"Loading processor: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name, language="en", task="transcribe"
    )
    return processor, feature_extractor, tokenizer


# ---------------------------------------------------------------------------
# 3. Dataset loading
# ---------------------------------------------------------------------------

def load_datasets(
    config: dict,
    processor: WhisperProcessor,
) -> Tuple[DatasetDict, Dataset]:
    """
    Load the prepared dataset from disk and apply feature extraction
    transform via ``map()``.
    """
    dataset_path = config["data"]["dataset_path"]
    logger.info(f"Loading dataset from {dataset_path}")

    dataset_dict = load_from_disk(dataset_path)

    sampling_rate = config["model"]["sampling_rate"]
    max_audio_length = config["data"].get("max_audio_length", 30.0)

    def prepare_batch(batch):
        audio = batch["audio"]
        waveform = audio["array"]

        # Truncate to max length
        max_samples = int(max_audio_length * sampling_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        input_features = processor.feature_extractor(
            waveform, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features[0]

        labels = processor.tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True
        ).input_ids[0]

        return {"input_features": input_features, "labels": labels}

    # Apply transform
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].map(
            prepare_batch,
            remove_columns=dataset_dict[split].column_names,
            desc=f"Processing {split} split",
        )

    test_dataset = dataset_dict[config["data"]["test_split"]]

    logger.info(
        f"Dataset loaded — "
        f"train: {len(dataset_dict['train'])}, "
        f"val: {len(dataset_dict['validation'])}, "
        f"test: {len(test_dataset)}"
    )

    return dataset_dict, test_dataset


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def compute_wer(
    preds: np.ndarray,
    labels: np.ndarray,
    tokenizer: WhisperTokenizer,
) -> Dict[str, float]:
    """Decode predictions and labels, compute WER."""
    wer_metric = hf_evaluate.load("wer")

    # Replace -100 with pad token id
    preds[preds == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean
    pred_str = [p.strip().lower() for p in pred_str]
    label_str = [l.strip().lower() for l in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ---------------------------------------------------------------------------
# 5. Trainer
# ---------------------------------------------------------------------------

def get_trainer(
    config: dict,
    model: WhisperForConditionalGeneration,
    tokenizer: WhisperTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    compute_metrics_fn: Callable,
    resume_checkpoint: Optional[str] = None,
) -> Seq2SeqTrainer:
    """Configure a ``Seq2SeqTrainer`` for Whisper fine-tuning."""
    training_config = config["training"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=training_config["output_dir"],
        max_steps=training_config["num_steps"],
        warmup_steps=training_config["warmup_steps"],
        learning_rate=float(training_config["learning_rate"]),
        optim=training_config.get("optimizer", "adamw_torch"),
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        fp16=training_config.get("fp16", True),
        logging_steps=training_config["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=training_config["eval_steps"],
        save_strategy="steps",
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=5,
        report_to=["tensorboard"],
        logging_dir=f"{training_config['output_dir']}/logs",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
    )

    return trainer


# ---------------------------------------------------------------------------
# 6. Train and save
# ---------------------------------------------------------------------------

def train_and_save(
    trainer: Seq2SeqTrainer,
    config: dict,
    use_lora: bool = False,
) -> None:
    """Run training and save the final model."""
    logger.info("Starting training...")

    trainer.train()

    output_dir = config["training"]["output_dir"]
    logger.info(f"Saving model to {output_dir}")

    if use_lora:
        # Save only the LoRA adapter weights
        trainer.model.save_pretrained(output_dir)
        logger.info("LoRA adapter weights saved")
    else:
        # Save full model
        trainer.save_model(output_dir)
        logger.info("Full model saved")

    # Save training state
    trainer.state.save_to_json(f"{output_dir}/trainer_state.json")

    logger.info("Training complete!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on Jamaican Patois"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint path (e.g., models/checkpoints/checkpoint-1000)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override LoRA from CLI
    if args.use_lora:
        config["lora"]["enabled"] = True

    use_lora = config["lora"]["enabled"]

    # Load processor
    model_name = config["model"]["name_or_path"]
    processor, feature_extractor, tokenizer = load_processor(model_name)

    # Load model
    model = load_model(
        model_name,
        use_lora=use_lora,
        lora_config=config.get("lora"),
        resume_checkpoint=args.resume,
    )

    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # Load datasets
    dataset_dict, test_dataset = load_datasets(config, processor)

    train_dataset = dataset_dict[config["data"]["train_split"]]
    val_dataset = dataset_dict[config["data"]["val_split"]]

    # Metrics
    def metrics_fn(preds, labels):
        return compute_wer(preds, labels, tokenizer)

    # Trainer
    trainer = get_trainer(
        config,
        model,
        processor.tokenizer,
        train_dataset,
        val_dataset,
        metrics_fn,
        resume_checkpoint=args.resume,
    )

    # Train
    train_and_save(trainer, config, use_lora=use_lora)

    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {json.dumps(test_results, indent=2)}")

    # Save test results
    output_dir = config["training"]["output_dir"]
    with open(f"{output_dir}/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Test results saved to {output_dir}/test_results.json")


if __name__ == "__main__":
    main()
