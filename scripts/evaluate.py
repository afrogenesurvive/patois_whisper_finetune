"""
Evaluation script for fine-tuned Whisper models on Jamaican Patois.

Computes WER (and optionally CER) on a test set with detailed error analysis.

Usage:
  python scripts/evaluate.py --model_path models/checkpoints --split test
  python scripts/evaluate.py --model_path models/checkpoints --split test --batch_size 16
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import evaluate as hf_evaluate
import librosa
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Load model and processor
# ---------------------------------------------------------------------------

def load_model_and_processor(
    model_path: str,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Load the fine-tuned model.

    Auto-detects LoRA by checking for ``adapter_config.json`` in
    ``model_path`` — if present, loads the base model first then
    applies the PEFT adapter.
    """
    model_path = Path(model_path)
    is_lora = (model_path / "adapter_config.json").exists()

    if is_lora:
        logger.info("LoRA adapter detected — loading base model + adapter")

        # Load processor from the adapter directory (has config)
        processor = WhisperProcessor.from_pretrained(str(model_path))

        # Load base model from the config's reference
        base_model_name = "openai/whisper-medium"
        # Try to read the base model name from adapter config
        adapter_config_path = model_path / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path", base_model_name)

        model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(model, str(model_path))
        logger.info(f"Loaded LoRA adapter on base model: {base_model_name}")
    else:
        logger.info("Loading full fine-tuned model")
        processor = WhisperProcessor.from_pretrained(str(model_path))
        model = WhisperForConditionalGeneration.from_pretrained(str(model_path))

    # Generation config
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    return model, processor


# ---------------------------------------------------------------------------
# 2. Batched transcription
# ---------------------------------------------------------------------------

def transcribe_batch(
    audio_paths: List[str],
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    batch_size: int = 8,
) -> List[str]:
    """
    Run inference in batches.

    Args:
        audio_paths: List of paths to 16kHz mono WAV files.
        model: The Whisper model.
        processor: The Whisper processor.
        device: 'cuda' or 'cpu'.
        batch_size: Inference batch size.

    Returns:
        List of transcribed text strings.
    """
    results: List[str] = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Transcribing"):
        batch_paths = audio_paths[i : i + batch_size]
        batch_audio = []

        for ap in batch_paths:
            waveform, _ = librosa.load(str(ap), sr=16000, mono=True)
            batch_audio.append(waveform)

        # Pad to same length
        max_len = max(len(w) for w in batch_audio)
        padded = np.zeros((len(batch_audio), max_len), dtype=np.float32)
        for j, w in enumerate(batch_audio):
            padded[j, : len(w)] = w

        input_features = processor.feature_extractor(
            padded, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                num_beams=5,
                max_length=128,
            )

        for j in range(len(batch_paths)):
            text = processor.tokenizer.decode(
                predicted_ids[j], skip_special_tokens=True
            )
            results.append(text)

    return results


# ---------------------------------------------------------------------------
# 3. Full evaluation
# ---------------------------------------------------------------------------

def evaluate(
    test_dataset: Dataset,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
    batch_size: int = 8,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Evaluate the model on a test dataset.

    Transcribes all test samples, computes WER/CER against ground truth,
    and returns results along with an error analysis table.

    Args:
        test_dataset: Dataset with 'audio' and 'text' columns.
        model: The Whisper model.
        processor: The Whisper processor.
        device: 'cuda' or 'cpu'.
        batch_size: Inference batch size.
        output_path: Optional path to save evaluation results JSON.

    Returns:
        Dict with 'wer', 'cer', and error analysis table.
    """
    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    # Collect audio paths and references
    audio_paths = []
    references = []
    for row in test_dataset:
        audio_paths.append(row["audio"]["path"])
        references.append(row["text"])

    logger.info(f"Evaluating on {len(audio_paths)} test samples")

    # Transcribe
    hypotheses = transcribe_batch(audio_paths, model, processor, device, batch_size)

    # Clean
    hypotheses_clean = [h.strip().lower() for h in hypotheses]
    references_clean = [r.strip().lower() for r in references]

    # Compute metrics
    wer = wer_metric.compute(predictions=hypotheses_clean, references=references_clean)
    cer = cer_metric.compute(predictions=hypotheses_clean, references=references_clean)

    logger.info(f"WER: {wer:.4f}")
    logger.info(f"CER: {cer:.4f}")

    # Build error analysis table
    error_table = []
    for hyp, ref, audio_path in zip(hypotheses, references, audio_paths):
        error_table.append({
            "audio": str(audio_path),
            "reference": ref,
            "hypothesis": hyp,
        })

    results = {
        "wer": wer,
        "cer": cer,
        "num_samples": len(error_table),
        "error_analysis": error_table,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Samples:    {len(error_table)}")
    print(f"  WER:        {wer:.4f}")
    print(f"  CER:        {cer:.4f}")
    print("=" * 70)

    if error_table:
        print("\nSample predictions (first 10):")
        print("-" * 70)
        for i, entry in enumerate(error_table[:10]):
            print(f"  [{i}] Ref:  {entry['reference']}")
            print(f"      Hyp:  {entry['hypothesis']}")
            print()

    # Save results
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results without full error table by default (can be large)
        results_file = output_path
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {results_file}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Whisper model on Jamaican Patois"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/dataset",
        help="Path to the prepared dataset (default: data/dataset)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/evaluation_results.json",
        help="Path to save evaluation results JSON (default: models/evaluation_results.json)",
    )

    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model, processor = load_model_and_processor(args.model_path)
    model = model.to(device)
    model.eval()

    # Load dataset
    dataset_dict = load_from_disk(args.dataset_path)
    test_dataset = dataset_dict[args.split]
    logger.info(f"Loaded {args.split} split with {len(test_dataset)} samples")

    # Evaluate
    evaluate(
        test_dataset,
        model,
        processor,
        device,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
