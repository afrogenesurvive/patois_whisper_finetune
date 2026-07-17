#!/usr/bin/env python3
"""
Whisper Fine-Tune GUI

A Gradio-based web interface for fine-tuning Whisper models on Jamaican
Patois audio.  Launches at http://127.0.0.1:7860

Usage:
    python gui/app.py
    python gui/app.py --port 7860 --share
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``scripts`` and ``gui``
# can be imported regardless of where the user runs the script from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr

from gui.tabs.data_tab import create_data_tab
from gui.tabs.train_tab import create_train_tab
from gui.tabs.evaluate_tab import create_evaluate_tab
from gui.tabs.infer_tab import create_infer_tab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_app() -> gr.Blocks:
    """Construct the Gradio application with all four tabs."""
    with gr.Blocks(
        title="Whisper Fine-Tune GUI",
        theme=gr.themes.Soft(),
        css="""
        footer { display: none !important; }
        .gradio-container { max-width: 1200px !important; }
        """,
    ) as app:
        gr.Markdown(
            "# 🎙 Whisper Fine-Tune GUI\n"
            "Fine-tune Whisper models for Jamaican Patois transcription. "
            "Upload audio, correct transcripts, train, evaluate, and transcribe — "
            "all from your browser."
        )

        # Create all four tabs
        create_data_tab()
        create_train_tab()
        create_evaluate_tab()
        create_infer_tab()

        gr.Markdown(
            "---\n"
            "**Whisper Fine-Tune GUI** | "
            "Built with [Gradio](https://gradio.app) | "
            "See `docs/training_manual.md` for full documentation"
        )

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Whisper Fine-Tune GUI"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve the GUI on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link (Gradio share)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    args = parser.parse_args()

    logger.info(f"Starting Whisper Fine-Tune GUI on port {args.port}")
    if args.share:
        logger.info("Share mode enabled — a public link will be generated.")

    app = build_app()
    app.launch(
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )


if __name__ == "__main__":
    main()
