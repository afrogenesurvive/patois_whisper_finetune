"""
Evaluate Tab — Run WER/CER evaluation on trained models with error analysis.
"""

import json
from pathlib import Path
from typing import Tuple

import gradio as gr

from gui.backend import run_evaluation
from gui.utils import list_checkpoints


def _refresh_checkpoints() -> list:
    """Refresh the checkpoint dropdown options."""
    ckpts = list_checkpoints()
    return gr.Dropdown(choices=ckpts, value=ckpts[0] if ckpts and "No" not in ckpts[0] else None)


def _on_evaluate(
    checkpoint: str,
    split: str,
    batch_size: int,
) -> Tuple[str, str, str, gr.Dataframe, str]:
    """Run evaluation and return results."""
    if not checkpoint or checkpoint == "No checkpoints found":
        return "—", "—", "—", gr.Dataframe(value=[[]]), "No checkpoint selected"

    model_path = f"models/checkpoints/{checkpoint}"
    success, results = run_evaluation(
        model_path=model_path,
        split=split,
        batch_size=batch_size,
    )

    if not success:
        error_msg = results.get("error", "Unknown error")
        return "—", "—", "—", gr.Dataframe(value=[[]]), f"Error: {error_msg}"

    wer = f"{results.get('wer', 0):.4f}"
    cer = f"{results.get('cer', 0):.4f}"
    samples = str(results.get("num_samples", 0))

    # Build error analysis table
    error_table = results.get("error_analysis", [])
    table_data = []
    for i, entry in enumerate(error_table[:100]):  # cap at 100 rows
        table_data.append([
            i + 1,
            entry.get("reference", ""),
            entry.get("hypothesis", ""),
        ])

    # Save timestamped results
    output_path = f"models/evaluation_results_{checkpoint}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return wer, cer, samples, gr.Dataframe(value=table_data), f"Results saved to {output_path}"


def create_evaluate_tab() -> gr.Tab:
    """Build the Gradio Evaluate tab."""
    checkpoints = list_checkpoints()
    default_ckpt = checkpoints[0] if checkpoints and "No" not in checkpoints[0] else None

    with gr.Tab("📊 Evaluate") as tab:
        gr.Markdown("## Evaluate a Trained Model")

        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(
                choices=checkpoints,
                value=default_ckpt,
                label="Model checkpoint",
                scale=3,
            )
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)

        with gr.Row():
            split = gr.Radio(
                choices=["test", "validation", "train"],
                value="test",
                label="Dataset split",
            )
            batch_size = gr.Number(
                value=8, label="Batch size", precision=0, minimum=1, maximum=64
            )

        eval_btn = gr.Button("▶ Run Evaluation", variant="primary")

        gr.Markdown("### Results")
        with gr.Row():
            wer_display = gr.Textbox(label="WER", value="—")
            cer_display = gr.Textbox(label="CER", value="—")
            samples_display = gr.Textbox(label="Samples", value="—")

        eval_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Error Analysis (Reference vs Hypothesis)")
        error_table = gr.Dataframe(
            headers=["#", "Reference", "Hypothesis"],
            label="Error analysis (first 100 samples)",
            interactive=False,
            wrap=True,
            column_widths=["5%", "45%", "50%"],
        )

        # ── Wire events ─────────────────────────────────────────────

        refresh_btn.click(
            fn=_refresh_checkpoints,
            outputs=[checkpoint_dropdown],
        )

        eval_btn.click(
            fn=_on_evaluate,
            inputs=[checkpoint_dropdown, split, batch_size],
            outputs=[wer_display, cer_display, samples_display, error_table, eval_status],
        )

    return tab
