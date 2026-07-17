"""
Data preparation script for Whisper fine-tuning on Jamaican Patois.

Pipeline:
  1. Resample raw audio to 16kHz mono WAV
  2. Generate pseudo-labels with a base Whisper model
  3. Load manually corrected transcripts
  4. Build train/val/test splits as a Hugging Face DatasetDict
  5. Save dataset info for inspection

Usage:
  # First pass: resample + generate pseudo-labels
  python scripts/prepare_data.py --raw_dir data/raw --model_size medium

  # Second pass: load corrected transcripts and build dataset
  python scripts/prepare_data.py --skip_pseudo
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Audio resampling
# ---------------------------------------------------------------------------

def resample_audio(
    raw_dir: Path,
    output_dir: Path,
    target_sr: int = 16000,
) -> List[Path]:
    """
    Walk `raw_dir`, load each audio file with librosa, resample to
    `target_sr` Hz mono, and save as 16-bit WAV to `output_dir`.

    Returns a list of processed file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    supported_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}

    audio_paths = []
    for ext in supported_extensions:
        audio_paths.extend(raw_dir.rglob(f"*{ext}"))

    if not audio_paths:
        logger.warning(f"No audio files found in {raw_dir}")
        return []

    processed = []
    for audio_path in tqdm(audio_paths, desc="Resampling audio"):
        try:
            # Load audio at original sample rate
            waveform, orig_sr = librosa.load(str(audio_path), sr=None, mono=True)

            # Resample if needed
            if orig_sr != target_sr:
                waveform = librosa.resample(
                    waveform, orig_sr=orig_sr, target_sr=target_sr
                )

            # Save as 16-bit WAV
            stem = audio_path.stem
            out_path = output_dir / f"{stem}.wav"
            sf.write(str(out_path), waveform, samplerate=target_sr, subtype="PCM_16")
            processed.append(out_path)

        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")

    logger.info(f"Resampled {len(processed)} / {len(audio_paths)} files to {target_sr} Hz")
    return processed


# ---------------------------------------------------------------------------
# 2. Pseudo-label generation
# ---------------------------------------------------------------------------

def generate_pseudo_labels(
    audio_paths: List[Path],
    model_size: str = "medium",
    batch_size: int = 8,
) -> Dict[str, str]:
    """
    Load a base Whisper model + processor from Hugging Face and transcribe
    each audio file.  Returns a dict mapping filename stem -> raw transcript.
    """
    model_name = f"openai/whisper-{model_size}"
    logger.info(f"Loading Whisper model: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )

    transcripts: Dict[str, str] = {}
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )

    # Process in batches
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Generating pseudo-labels"):
        batch_paths = audio_paths[i : i + batch_size]
        batch_audio = []

        for ap in batch_paths:
            waveform, sr = librosa.load(str(ap), sr=16000, mono=True)
            batch_audio.append(waveform)

        # Pad to same length for batched inference
        max_len = max(len(w) for w in batch_audio)
        padded = torch.zeros(len(batch_audio), max_len)
        for j, w in enumerate(batch_audio):
            padded[j, : len(w)] = torch.tensor(w)

        input_features = processor.feature_extractor(
            padded.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
            )

        for j, ap in enumerate(batch_paths):
            text = processor.tokenizer.decode(
                predicted_ids[j], skip_special_tokens=True
            )
            transcripts[ap.stem] = text

    return transcripts


# ---------------------------------------------------------------------------
# 3. Load corrected transcripts
# ---------------------------------------------------------------------------

def load_corrected_transcripts(transcript_dir: Path) -> Dict[str, str]:
    """
    Load manually corrected transcript files from `transcript_dir`.

    Supports:
      - JSON:  {"filename": "transcript text", ...}
      - CSV:   columns ``filename``, ``transcript``
    """
    transcripts: Dict[str, str] = {}

    if not transcript_dir.exists():
        logger.warning(f"Transcript directory does not exist: {transcript_dir}")
        return transcripts

    # JSON files
    for json_path in transcript_dir.glob("*.json"):
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            transcripts.update(data)
        elif isinstance(data, list):
            for item in data:
                if "filename" in item and "transcript" in item:
                    transcripts[item["filename"]] = item["transcript"]
        logger.info(f"Loaded {len(data)} transcripts from {json_path.name}")

    # CSV files
    for csv_path in transcript_dir.glob("*.csv"):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                fname = row.get("filename", "")
                text = row.get("transcript", "")
                if fname and text:
                    # Strip extension if present for consistency
                    transcripts[Path(fname).stem] = text
                    count += 1
        logger.info(f"Loaded {count} transcripts from {csv_path.name}")

    logger.info(f"Total corrected transcripts loaded: {len(transcripts)}")
    return transcripts


# ---------------------------------------------------------------------------
# 4. Build Hugging Face DatasetDict
# ---------------------------------------------------------------------------

def create_dataset_dict(
    audio_dir: Path,
    transcripts: Dict[str, str],
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> DatasetDict:
    """
    Build a Hugging Face ``DatasetDict`` with columns ``audio`` (path) and
    ``text`` (transcript).  Split using ``val_split`` and ``test_split``
    fractions, then save a preprocessed copy to ``data/dataset/`` for caching.
    """
    audio_files = list(audio_dir.glob("*.wav"))
    logger.info(f"Found {len(audio_files)} processed audio files")

    # Match audio files to transcripts
    data_rows = []
    missing = 0
    for audio_path in audio_files:
        stem = audio_path.stem
        if stem in transcripts:
            data_rows.append({
                "audio": str(audio_path),
                "text": transcripts[stem],
            })
        else:
            missing += 1

    if missing:
        logger.warning(
            f"{missing} audio files have no matching transcript — they will be skipped"
        )

    if not data_rows:
        raise ValueError(
            "No matching (audio, transcript) pairs found. "
            "Ensure transcript filenames match audio filenames (without extension)."
        )

    logger.info(f"Matched {len(data_rows)} audio-transcript pairs")

    # Create dataset
    dataset = Dataset.from_list(data_rows)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split
    val_fraction = val_split / (1.0 - test_split)
    train_test_split = dataset.train_test_split(
        test_size=val_split + test_split, seed=42
    )
    test_val_split = train_test_split["test"].train_test_split(
        test_size=val_fraction, seed=42
    )

    dataset_dict = DatasetDict({
        "train": train_test_split["train"],
        "validation": test_val_split["test"],
        "test": test_val_split["train"],
    })

    logger.info(
        f"Split sizes — train: {len(dataset_dict['train'])}, "
        f"validation: {len(dataset_dict['validation'])}, "
        f"test: {len(dataset_dict['test'])}"
    )

    return dataset_dict


# ---------------------------------------------------------------------------
# 5. Prepare dataset (feature extraction)
# ---------------------------------------------------------------------------

def prepare_dataset(
    dataset: Dataset,
    processor: WhisperProcessor,
    max_length: int = 30,
) -> Dataset:
    """
    Apply the Whisper feature extractor to the audio column and tokenizer
    to the text column.  Returns a transformed dataset ready for training.
    """
    sampling_rate = processor.feature_extractor.sampling_rate

    def transform(batch):
        audio = batch["audio"]

        # Load and pad/truncate audio
        waveform = audio["array"]
        if len(waveform) > max_length * sampling_rate:
            waveform = waveform[: max_length * sampling_rate]
        elif len(waveform) == 0:
            waveform = [0.0]

        input_features = processor.feature_extractor(
            waveform, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features[0]

        labels = processor.tokenizer(
            batch["text"], return_tensors="pt", padding=True
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
        }

    return dataset.map(transform, remove_columns=["audio"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def save_dataset_info(dataset_dict: DatasetDict, output_dir: Path) -> None:
    """Save a JSON file with dataset metadata for inspection."""
    info = {
        "splits": {
            name: {
                "num_samples": len(ds),
                "columns": list(ds.column_names),
            }
            for name, ds in dataset_dict.items()
        },
        "sampling_rate": 16000,
        "example_rows": {},
    }

    for name, ds in dataset_dict.items():
        examples = []
        for i in range(min(3, len(ds))):
            row = ds[i]
            examples.append({
                "audio_path": row.get("audio", str(row.get("input_features", "N/A"))),
                "text": row.get("text", "N/A"),
            })
        info["example_rows"][name] = examples

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Dataset info saved to {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio data for Whisper fine-tuning on Jamaican Patois"
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with raw audio files (default: data/raw)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for resampled audio (default: data/processed)",
    )
    parser.add_argument(
        "--transcript_dir",
        type=Path,
        default=Path("data/transcripts"),
        help="Directory with corrected transcript files (default: data/transcripts)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/dataset"),
        help="Output directory for HF dataset artifacts (default: data/dataset)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size for pseudo-labeling (default: medium)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction of data for test (default: 0.1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for pseudo-label generation (default: 8)",
    )
    parser.add_argument(
        "--skip_pseudo",
        action="store_true",
        help="Skip pseudo-label generation and load existing corrected transcripts",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Resample audio
    # -----------------------------------------------------------------------
    logger.info("Step 1: Resampling audio to 16kHz mono")
    processed_paths = resample_audio(args.raw_dir, args.output_dir)

    if not processed_paths:
        logger.error("No audio files were processed. Check your --raw_dir.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Generate pseudo-labels (first pass only)
    # -----------------------------------------------------------------------
    if not args.skip_pseudo:
        logger.info("Step 2: Generating pseudo-labels with base Whisper model")
        transcripts = generate_pseudo_labels(
            processed_paths,
            model_size=args.model_size,
            batch_size=args.batch_size,
        )

        print("\n" + "=" * 70)
        print("PSEUDO-LABEL TRANSCRIPTS GENERATED")
        print("=" * 70)
        for stem, text in list(transcripts.items())[:10]:
            print(f"  {stem}: {text}")
        if len(transcripts) > 10:
            print(f"  ... and {len(transcripts) - 10} more")
        print("=" * 70)
        print(
            "Instructions:\n"
            f"  1. Correct these transcripts in: {args.transcript_dir}/\n"
            "     (save as JSON: {\"filename\": \"corrected text\", ...}\n"
            "      or CSV: filename,transcript)\n"
            "  2. Re-run with --skip_pseudo to build the dataset\n"
        )
    else:
        # -------------------------------------------------------------------
        # Step 3: Load corrected transcripts & build dataset
        # -------------------------------------------------------------------
        logger.info("Step 3: Loading corrected transcripts")
        corrected = load_corrected_transcripts(args.transcript_dir)

        if not corrected:
            logger.error(
                "No corrected transcripts found. "
                f"Place transcript files in {args.transcript_dir}/ and try again."
            )
            sys.exit(1)

        logger.info("Step 4: Building dataset splits")
        dataset_dict = create_dataset_dict(
            args.output_dir,
            corrected,
            val_split=args.val_split,
            test_split=args.test_split,
        )

        # Save dataset to disk
        dataset_dict.save_to_disk(str(args.dataset_dir))
        logger.info(f"Dataset saved to {args.dataset_dir}")

        # Save info
        save_dataset_info(dataset_dict, args.dataset_dir)

        print("\n" + "=" * 70)
        print("DATASET READY FOR TRAINING")
        print("=" * 70)
        print(f"  Train:      {len(dataset_dict['train'])} samples")
        print(f"  Validation: {len(dataset_dict['validation'])} samples")
        print(f"  Test:       {len(dataset_dict['test'])} samples")
        print(f"  Location:   {args.dataset_dir}")
        print("=" * 70)
        print("\nNext step: python scripts/train.py --config config.yaml\n")


if __name__ == "__main__":
    main()
