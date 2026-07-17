"""
Infer Tab — Transcribe single or multiple audio files with a trained model.
"""

import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from gui.backend import run_transcription
from gui.utils import list_checkpoints


def _refresh_checkpoints() -> list:
    """Refresh the checkpoint dropdown options."""
    ckpts = list_checkpoints()
    return gr.Dropdown(
        choices=ckpts,
        value=ckpts[0] if ckpts and "No" not in ckpts[0] else None,
    )


def _on_transcribe(
    audio: Optional[str],
    checkpoint: str,
    output_file: Optional[str],
) -> Tuple[str, str, Optional[str]]:
    """Transcribe a single audio file."""
    if not audio:
        return "", "No audio file provided.", None
    if not checkpoint or checkpoint == "No checkpoints found":
        return "", "No model checkpoint selected.", None

    model_path = f"models/checkpoints/{checkpoint}"
    success, result = run_transcription(audio, model_path)

    if success:
        # Save to file if requested
        saved_path = None
        if output_file:
            out = Path(output_file)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                f.write(result + "\n")
            saved_path = str(out)
        return result, "✅ Transcription complete!", saved_path
    else:
        return "", f"❌ {result}", None


def _on_batch_transcribe(
    files: List[str],
    checkpoint: str,
) -> Tuple[str, Optional[str]]:
    """Transcribe multiple audio files and return a ZIP."""
    if not files:
        return "No files uploaded.", None
    if not checkpoint or checkpoint == "No checkpoints found":
        return "No model checkpoint selected.", None

    model_path = f"models/checkpoints/{checkpoint}"
    results = []

    for filepath in files:
        stem = Path(filepath).stem
        success, text = run_transcription(filepath, model_path)
        if success:
            results.append((stem, text))
        else:
            results.append((stem, f"[ERROR] {text}"))

    # Create a ZIP with individual .txt files
    zip_path = os.path.join(tempfile.gettempdir(), "transcriptions.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem, text in results:
            zf.writestr(f"{stem}.txt", text + "\n")

    summary = "\n\n".join(
        f"--- {stem} ---\n{text}" for stem, text in results
    )

    return summary, zip_path


def create_infer_tab() -> gr.Tab:
    """Build the Gradio Infer tab."""
    checkpoints = list_checkpoints()
    default_ckpt = checkpoints[0] if checkpoints and "No" not in checkpoints[0] else None

    with gr.Tab("🎙 Infer") as tab:
        gr.Markdown("## Transcribe Audio with a Trained Model")

        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(
                choices=checkpoints,
                value=default_ckpt,
                label="Model checkpoint",
                scale=3,
            )
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)

        gr.Markdown("### Single File Transcription")

        with gr.Row():
            audio_input = gr.Audio(
                label="Upload or record audio",
                type="filepath",
                sources=["upload", "microphone"],
                scale=3,
            )
            with gr.Column(scale=1):
                output_filename = gr.Textbox(
                    label="Save to file (optional)",
                    placeholder="e.g., result.txt",
                )
                transcribe_btn = gr.Button("▶ Transcribe", variant="primary")

        transcription_output = gr.Textbox(
            label="Transcription",
            lines=5,
            interactive=False,
        )
        infer_status = gr.Textbox(label="Status", interactive=False)
        download_single = gr.File(label="Download transcription", visible=False)

        gr.Markdown("### Batch Transcription")
        with gr.Row():
            batch_files = gr.File(
                label="Upload multiple audio files",
                file_count="multiple",
                file_types=["audio"],
                scale=3,
            )
            batch_btn = gr.Button("▶ Transcribe All", variant="primary", scale=1)

        batch_output = gr.Textbox(
            label="Batch results",
            lines=8,
            interactive=False,
        )
        download_batch = gr.File(label="Download all as ZIP")

        # ── Wire events ─────────────────────────────────────────────

        refresh_btn.click(
            fn=_refresh_checkpoints,
            outputs=[checkpoint_dropdown],
        )

        transcribe_btn.click(
            fn=_on_transcribe,
            inputs=[audio_input, checkpoint_dropdown, output_filename],
            outputs=[transcription_output, infer_status, download_single],
        ).then(
            fn=lambda path: gr.update(value=path, visible=path is not None),
            inputs=[download_single],
            outputs=[download_single],
        )

        batch_btn.click(
            fn=_on_batch_transcribe,
            inputs=[batch_files, checkpoint_dropdown],
            outputs=[batch_output, download_batch],
        )

    return tab
