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
from gui.tabs.guide_tab import create_guide_tab
from gui.tabs.infer_tab import create_infer_tab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_app() -> gr.Blocks:
    """Construct the Gradio application with all four tabs."""

    # Install the shared log buffer so Gradio-side logs are captured too
    from gui.utils import log_buffer
    logging.getLogger().addHandler(log_buffer)

    with gr.Blocks(
        title="Whisper Fine-Tune GUI",
        theme=gr.themes.Soft(),
        css="""
        footer { display: none !important; }
        .gradio-container { max-width: 1200px !important; }

        /* Floating "Quit App" button, pinned to the top-left corner */
        #quit-app-btn {
            position: fixed !important;
            top: 12px !important;
            left: 12px !important;
            z-index: 9999 !important;
            border: none;
            border-radius: 8px;
            background: rgba(60, 60, 60, 0.85);
            color: #fff;
            padding: 8px 14px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        }
        #quit-app-btn:hover { background: #c0392b; }

        /* Quit confirmation overlay */
        .quit-overlay {
            position: fixed !important;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.45);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 100000 !important;
        }
        .quit-dialog {
            background: #fff;
            border-radius: 12px;
            padding: 24px 28px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
            max-width: 360px;
            text-align: center;
        }
        .quit-dialog h3 { margin: 0 0 8px; font-size: 17px; }
        .quit-dialog p { color: #666; margin: 0 0 18px; font-size: 14px; }
        .quit-actions { display: flex; gap: 10px; justify-content: center; }
        .quit-actions button {
            border: none; border-radius: 8px;
            padding: 8px 18px; font-size: 13px; font-weight: 600; cursor: pointer;
        }
        .quit-cancel { background: #e2e2e2; color: #333; }
        .quit-cancel:hover { background: #d0d0d0; }
        .quit-confirm { background: #c0392b; color: #fff; }
        .quit-confirm:hover { background: #a93226; }
        """,
    ) as app:
        # Floating "Quit App" button — pinned top-left. Quits via the
        # pywebview JS bridge (only wired when running inside the .app;
        # a harmless no-op when opened in a plain browser).
        gr.HTML(
            """
            <button id="quit-app-btn"
                    title="Quit the application"
                    onclick="document.getElementById('quit-overlay').style.display='flex'">
              ✕ Quit App
            </button>
            <div id="quit-overlay" class="quit-overlay" style="display:none;">
              <div class="quit-dialog">
                <h3>Quit Whisper Fine-Tune GUI?</h3>
                <p>Any in-progress training or unsaved work will be lost.</p>
                <div class="quit-actions">
                  <button id="quit-cancel" class="quit-cancel"
                          onclick="document.getElementById('quit-overlay').style.display='none'">Cancel</button>
                  <button id="quit-confirm" class="quit-confirm"
                          onclick="if (window.pywebview && window.pywebview.api && window.pywebview.api.quit_app) { pywebview.api.quit_app(); }">Quit</button>
                </div>
              </div>
            </div>
            """
        )

        gr.Markdown(
            "# 🎙 Whisper Fine-Tune GUI\n"
            "Fine-tune Whisper models for Jamaican Patois transcription. "
            "Upload audio, correct transcripts, train, evaluate, and transcribe — "
            "all from your browser."
        )

        # Create all five tabs
        # Pass the app (Blocks) instance so tabs can register load events
        create_data_tab(app)
        create_train_tab(app)
        create_evaluate_tab()
        create_guide_tab()
        create_infer_tab()

        gr.Markdown(
            "---\n"
            "**Whisper Fine-Tune GUI** | "
            "Built with [Gradio](https://gradio.app) | "
            "See `docs/training_manual.md` for full documentation"
        )

        # ── App Logs accordion (Option C: in-app log viewer) ────────────
        with gr.Accordion("📋 App Logs", open=False):
            app_logs = gr.Textbox(
                label="Application log output",
                lines=10,
                max_lines=20,
                interactive=False,
            )

        # Periodic log refresh every 3 seconds
        from gui.utils import get_app_logs
        _log_timer = gr.Timer(3)
        _log_timer.tick(
            fn=lambda: "\n".join(get_app_logs(40)),
            outputs=[app_logs],
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
