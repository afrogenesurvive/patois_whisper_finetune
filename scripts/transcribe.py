"""
Inference script for a fine-tuned Whisper model on Jamaican Patois.

Transcribes a single audio file (any format) using the fine-tuned model.

Usage:
  python scripts/transcribe.py --model_path models/checkpoints --audio path/to/audio.wav
  python scripts/transcribe.py --model_path models/checkpoints --audio clip.mp3 --output result.txt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import librosa
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model_and_processor(
    model_path: str,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Load the fine-tuned model.

    Auto-detects LoRA by checking for ``adapter_config.json`` —
    if present, loads base model first then applies the PEFT adapter.
    """
    model_path = Path(model_path)
    is_lora = (model_path / "adapter_config.json").exists()

    if is_lora:
        logger.info("LoRA adapter detected — loading base model + adapter")
        processor = WhisperProcessor.from_pretrained(str(model_path))

        base_model_name = "openai/whisper-medium"
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

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    return model, processor


def transcribe_file(
    audio_path: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str,
) -> str:
    """
    Load a single audio file, resample to 16kHz mono, run Whisper
    inference, and return the transcribed text.

    Args:
        audio_path: Path to the audio file (any format supported by librosa).
        model: The Whisper model.
        processor: The Whisper processor.
        device: 'cuda' or 'cpu'.

    Returns:
        Transcribed text string.
    """
    logger.info(f"Loading audio: {audio_path}")

    # Load and resample
    waveform, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    if len(waveform) == 0:
        raise ValueError(f"Audio file {audio_path} is empty or could not be read")

    # Compute input features
    input_features = processor.feature_extractor(
        waveform, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    # Generate
    logger.info("Running inference...")
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            num_beams=5,
            max_length=128,
        )

    # Decode
    transcription = processor.tokenizer.decode(
        predicted_ids[0], skip_special_tokens=True
    )

    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with a fine-tuned Whisper model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to the audio file to transcribe",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the transcription (printed to stdout if omitted)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Defaults to auto-detect.",
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.audio).exists():
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

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

    # Transcribe
    transcription = transcribe_file(args.audio, model, processor, device)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(transcription + "\n")
        logger.info(f"Transcription saved to {output_path}")
    else:
        print("\n" + "=" * 70)
        print("TRANSCRIPTION")
        print("=" * 70)
        print(transcription)
        print("=" * 70)


if __name__ == "__main__":
    main()
