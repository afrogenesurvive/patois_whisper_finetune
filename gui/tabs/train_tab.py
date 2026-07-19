"""
Train Tab — Configure, start, stop, and monitor model training.

Provides a configuration form (populated from ``config.yaml``), start/stop
buttons, live WER/Loss charts via TensorBoard polling, and a console log.
"""

import threading
from pathlib import Path
from typing import Tuple

import gradio as gr
import plotly.graph_objects as go
import yaml

from gui.backend import start_training, stop_training
from gui.state import training_state
from gui.utils import format_time, load_config, read_tb_scalar


def _load_config_form() -> dict:
    """Load current config.yaml into a dict for form population."""
    config = load_config()
    if not config:
        return {}
    return config


def _get_status() -> Tuple[str, str, str]:
    """Get current training status text."""
    state = training_state.snapshot()
    if not state["running"]:
        return "⏸ Idle", "—", "—"
    elapsed = format_time(state["elapsed_seconds"])
    step_info = f"Step {state['current_step']}/{state['total_steps']}"
    return f"● Running ({elapsed})", step_info, ""


def _get_chart(tag: str, title: str, y_label: str) -> go.Figure:
    """Build a Plotly chart from TensorBoard scalar data."""
    log_dir = "models/checkpoints/logs"
    steps, values = read_tb_scalar(log_dir, tag)

    fig = go.Figure()
    if steps and values:
        fig.add_trace(go.Scatter(x=steps, y=values, mode="lines", name=title))
        fig.update_layout(
            title=title,
            xaxis_title="Step",
            yaxis_title=y_label,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
        )
    else:
        fig.add_annotation(
            text="Waiting for data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(
            title=title, template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
        )
    return fig


def _get_loss_chart() -> go.Figure:
    return _get_chart("train/loss", "Training Loss", "Loss")


def _get_wer_chart() -> go.Figure:
    return _get_chart("eval/wer", "Validation WER", "WER")


def _on_start() -> str:
    """Handle start training button."""
    return start_training()


def _on_stop() -> str:
    """Handle stop training button."""
    return stop_training()


def create_train_tab(app: gr.Blocks) -> gr.Tab:
    """Build the Gradio Train tab."""
    config = _load_config_form()
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    with gr.Tab("🎓 Train") as tab:
        gr.Markdown("## Training Configuration")

        with gr.Row():
            model_name = gr.Dropdown(
                choices=[
                    "openai/whisper-tiny",
                    "openai/whisper-base",
                    "openai/whisper-small",
                    "openai/whisper-medium",
                    "openai/whisper-large",
                    "openai/whisper-large-v2",
                    "openai/whisper-large-v3",
                ],
                value=model_cfg.get("name_or_path", "openai/whisper-medium"),
                label="Base Model",
                scale=2,
            )
            training_mode = gr.Radio(
                choices=["Full Fine-Tune", "LoRA"],
                value="LoRA" if lora_cfg.get("enabled", False) else "Full Fine-Tune",
                label="Training Mode",
                scale=1,
            )

        with gr.Accordion("Advanced Hyperparameters", open=False):
            with gr.Row():
                num_steps = gr.Number(
                    value=train_cfg.get("num_steps", 4000),
                    label="Training steps",
                    precision=0,
                    minimum=100,
                    maximum=20000,
                )
                learning_rate = gr.Number(
                    value=train_cfg.get("learning_rate", 1.0e-5),
                    label="Learning rate",
                    minimum=1e-7,
                    maximum=1e-3,
                )
                warmup_steps = gr.Number(
                    value=train_cfg.get("warmup_steps", 500),
                    label="Warmup steps",
                    precision=0,
                    minimum=0,
                )
            with gr.Row():
                batch_size = gr.Number(
                    value=train_cfg.get("per_device_train_batch_size", 8),
                    label="Batch size (per device)",
                    precision=0,
                    minimum=1,
                    maximum=64,
                )
                grad_accum = gr.Number(
                    value=train_cfg.get("gradient_accumulation_steps", 2),
                    label="Gradient accumulation",
                    precision=0,
                    minimum=1,
                    maximum=16,
                )
                fp16 = gr.Checkbox(
                    value=train_cfg.get("fp16", True),
                    label="FP16 (mixed precision)",
                )

            with gr.Row():
                lora_r = gr.Number(
                    value=lora_cfg.get("r", 8),
                    label="LoRA rank (r)",
                    precision=0,
                    minimum=1,
                    maximum=64,
                )
                lora_alpha = gr.Number(
                    value=lora_cfg.get("lora_alpha", 32),
                    label="LoRA alpha",
                    precision=0,
                    minimum=1,
                    maximum=128,
                )
                lora_dropout = gr.Number(
                    value=lora_cfg.get("lora_dropout", 0.05),
                    label="LoRA dropout",
                    minimum=0.0,
                    maximum=0.5,
                )

        with gr.Row():
            save_btn = gr.Button("💾 Save Configuration", variant="secondary")
            save_status = gr.Textbox(label="", interactive=False, visible=False)

        gr.Markdown("## Controls")
        with gr.Row():
            start_btn = gr.Button("▶ Start Training", variant="primary", scale=2)
            stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)

        with gr.Row():
            status_text = gr.Textbox(label="Status", value="⏸ Idle", interactive=False)
            step_text = gr.Textbox(label="Progress", value="—", interactive=False)

        gr.Markdown("## Live Metrics")
        with gr.Row():
            loss_chart = gr.Plot(label="Training Loss", value=_get_loss_chart())
            wer_chart = gr.Plot(label="Validation WER", value=_get_wer_chart())

        gr.Markdown("## Console Log")
        console_log = gr.Textbox(
            label="Training output",
            lines=10,
            max_lines=20,
            interactive=False,
        )

        # ── Helpers to save config ──────────────────────────────────

        def _on_save_config(
            model: str,
            mode: str,
            steps: int,
            lr: float,
            warmup: int,
            batch: int,
            accum: int,
            fp16_val: bool,
            lora_r_val: int,
            lora_alpha_val: int,
            lora_dropout_val: float,
        ) -> str:
            config = {
                "model": {
                    "name_or_path": model,
                    "language": "en",
                    "task": "transcribe",
                    "sampling_rate": 16000,
                },
                "training": {
                    "output_dir": "models/checkpoints",
                    "num_steps": steps,
                    "warmup_steps": warmup,
                    "learning_rate": float(lr),
                    "optimizer": "adamw_torch",
                    "per_device_train_batch_size": batch,
                    "gradient_accumulation_steps": accum,
                    "per_device_eval_batch_size": batch,
                    "fp16": fp16_val,
                    "logging_steps": 50,
                    "eval_steps": 200,
                    "save_steps": 500,
                    "save_total_limit": 3,
                    "metric_for_best_model": "wer",
                    "greater_is_better": False,
                    "load_best_model_at_end": True,
                },
                "lora": {
                    "enabled": mode == "LoRA",
                    "r": lora_r_val,
                    "lora_alpha": lora_alpha_val,
                    "target_modules": ["q_proj", "v_proj"],
                    "lora_dropout": lora_dropout_val,
                    "bias": "none",
                },
                "data": {
                    "dataset_path": "data/dataset",
                    "train_split": "train",
                    "val_split": "validation",
                    "test_split": "test",
                    "max_audio_length": 30.0,
                },
            }
            from gui.utils import save_config
            save_config("config.yaml", config)
            return "✅ Configuration saved to config.yaml"

        # ── Periodic update function ────────────────────────────────

        def _periodic_update():
            state = training_state.snapshot()
            if state["running"]:
                elapsed = format_time(state["elapsed_seconds"])
                step_info = f"Step {state['current_step']}/{state['total_steps']}"
                status = f"● Running ({elapsed})"
                logs = "\n".join(state["logs"][-15:])
            else:
                status = "⏸ Idle"
                step_info = "—"
                logs = "\n".join(training_state.logs[-15:])

            return (
                status,
                step_info,
                _get_loss_chart(),
                _get_wer_chart(),
                logs,
            )

        # ── Wire events ─────────────────────────────────────────────

        save_btn.click(
            fn=_on_save_config,
            inputs=[
                model_name, training_mode, num_steps, learning_rate,
                warmup_steps, batch_size, grad_accum, fp16,
                lora_r, lora_alpha, lora_dropout,
            ],
            outputs=[save_status],
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[save_status],
        )

        start_btn.click(
            fn=_on_start,
            inputs=[],
            outputs=[status_text],
        )

        stop_btn.click(
            fn=_on_stop,
            inputs=[],
            outputs=[status_text],
        )

        # Periodic refresh every 2 seconds
        timer = gr.Timer(2)
        timer.tick(
            fn=_periodic_update,
            outputs=[status_text, step_text, loss_chart, wer_chart, console_log],
        )

    return tab
