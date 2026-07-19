"""
Data Tab — Audio upload, pseudo-labeling, transcript correction, and dataset
building.

Uses ``gui.backend`` to call ``scripts.prepare_data`` functions.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

from gui.backend import (
    get_corrections,
    run_build_dataset,
    run_pseudo_label,
    run_resample,
    save_correction,
)
from gui.utils import list_checkpoints


def _list_audio_files() -> List[str]:
    """Return sorted list of stems for processed audio files."""
    proc = Path("data/processed")
    if not proc.exists():
        return []
    return sorted(p.stem for p in proc.glob("*.wav"))


def _build_file_table(audio_stems: List[str]) -> List[List[str]]:
    """
    Build a table with columns: Play button, filename, pseudo-label,
    corrected text (if any), and edit/save buttons.
    """
    corrections = get_corrections()
    rows = []
    for stem in audio_stems:
        pseudo = "—"
        corrected = corrections.get(stem, "")
        rows.append([stem, corrected or "✏ Not yet corrected"])
    return rows


def _on_upload(files: List[str]) -> Tuple[str, List[List[str]]]:
    """Handle file upload: save to data/raw, then resample."""
    if not files:
        return "No files uploaded.", []

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Move uploaded files to data/raw/
    for f in files:
        src = Path(f)
        dst = raw_dir / src.name
        if not dst.exists():
            os.rename(str(src), str(dst))

    success, msg = run_resample()
    stems = _list_audio_files()
    return msg, [[s, get_corrections().get(s, "✏ Not yet corrected")] for s in stems]


def _on_generate_pseudo(model_size: str) -> Tuple[str, List[List[str]]]:
    """Generate pseudo-labels and rebuild the file table."""
    success, msg, transcripts = run_pseudo_label(model_size=model_size)
    stems = _list_audio_files()
    corrections = get_corrections()
    table = []
    for stem in stems:
        pseudo = transcripts.get(stem, "—")
        corrected = corrections.get(stem, "")
        table.append([stem, corrected or f"(pseudo: {pseudo[:60]}…)"])

    # Also display raw transcripts for the user
    transcript_summary = "\n".join(
        f"  {k}: {v[:80]}" for k, v in list(transcripts.items())[:20]
    )
    if len(transcripts) > 20:
        transcript_summary += f"\n  ... and {len(transcripts) - 20} more"

    return (
        f"{msg}\n\nPseudo-labels generated. Click ✏ to correct each file.",
        table,
    )


def _on_correct(stem: str, text: str) -> str:
    """Save a corrected transcript."""
    return save_correction(stem, text)


def _on_build_dataset(val_split: float, test_split: float) -> Tuple[str, str, str, str]:
    """Build the Hugging Face dataset."""
    success, msg, sizes = run_build_dataset(
        val_split=val_split, test_split=test_split
    )
    if success:
        return (
            msg,
            str(sizes.get("train", "—")),
            str(sizes.get("validation", "—")),
            str(sizes.get("test", "—")),
        )
    return msg, "—", "—", "—"


def _refresh_file_list() -> List:
    """Refresh the file table (called periodically)."""
    stems = _list_audio_files()
    corrections = get_corrections()
    return [[s, corrections.get(s, "✏ Not yet corrected")] for s in stems]


def create_data_tab(app: gr.Blocks) -> gr.Tab:
    """Build the Gradio Data tab."""
    with gr.Tab("🗂 Data") as tab:
        gr.Markdown(
            "## 1. Upload Audio\n"
            "Drop audio files (WAV, MP3, FLAC, M4A, OGG) or record directly."
        )

        with gr.Row():
            upload = gr.File(
                label="Drop audio files here",
                file_count="multiple",
                file_types=["audio"],
                scale=3,
            )
            record = gr.Audio(
                label="Or record audio",
                type="filepath",
                sources=["microphone"],
                scale=2,
            )

        upload_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("## 2. Generate Pseudo-Labels")

        with gr.Row():
            model_size = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                value="medium",
                label="Whisper model size",
                scale=1,
            )
            gen_btn = gr.Button("🎙 Generate Pseudo-Labels", variant="primary", scale=2)

        pseudo_status = gr.Textbox(label="Pseudo-label status", interactive=False, lines=5)

        gr.Markdown("## 3. Correct Transcripts")

        # File table: click a row to edit the correction
        file_table = gr.Dataframe(
            headers=["Filename", "Transcript Status"],
            label="Audio files (click row to edit correction)",
            interactive=False,
            wrap=True,
            column_widths=["30%", "70%"],
        )

        with gr.Row():
            correction_stem = gr.Textbox(
                label="Filename to correct", scale=1, placeholder="e.g., clip_001"
            )
            correction_text = gr.Textbox(
                label="Corrected transcript",
                scale=2,
                lines=2,
                placeholder="Type the corrected English transcript here...",
            )
            save_btn = gr.Button("💾 Save Correction", variant="secondary", scale=1)

        correction_status = gr.Textbox(label="Correction status", interactive=False)

        gr.Markdown("## 4. Build Dataset")

        with gr.Row():
            val_split = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Validation split")
            test_split = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Test split")

        build_btn = gr.Button("🚀 Build Dataset", variant="primary")

        with gr.Row():
            train_count = gr.Textbox(label="Train samples", interactive=False)
            val_count = gr.Textbox(label="Validation samples", interactive=False)
            test_count = gr.Textbox(label="Test samples", interactive=False)

        build_status = gr.Textbox(label="Dataset build status", interactive=False)

        # ── Event wiring ────────────────────────────────────────────

        # Upload triggers resample
        upload.change(
            fn=_on_upload,
            inputs=[upload],
            outputs=[upload_status, file_table],
        )

        # Record also triggers resample
        record.stop_recording(
            fn=lambda x: _on_upload([x]) if x else ("No recording.", []),
            inputs=[record],
            outputs=[upload_status, file_table],
        )

        # Generate pseudo-labels
        gen_btn.click(
            fn=_on_generate_pseudo,
            inputs=[model_size],
            outputs=[pseudo_status, file_table],
        )

        # Save correction
        save_btn.click(
            fn=_on_correct,
            inputs=[correction_stem, correction_text],
            outputs=[correction_status],
        ).then(
            fn=_refresh_file_list,
            outputs=[file_table],
        )

        # Build dataset
        build_btn.click(
            fn=_on_build_dataset,
            inputs=[val_split, test_split],
            outputs=[build_status, train_count, val_count, test_count],
        )

        # Refresh file list periodically (every 10s)
        timer = gr.Timer(10)
        timer.tick(
            fn=_refresh_file_list,
            outputs=[file_table],
        )

    return tab
