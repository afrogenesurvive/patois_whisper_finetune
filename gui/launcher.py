#!/usr/bin/env python3
"""
Native macOS app launcher for the Whisper Fine-Tune GUI.

Wraps the Gradio web interface in a ``pywebview`` native macOS window so
the app can be launched from Finder / Dock instead of the terminal.

Usage (direct — fast dev loop):
    python gui/launcher.py

Build a standalone .app bundle:
    python setup_app.py py2app
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Ensure the project root is on sys.path so ``gui`` and ``scripts``
# can be imported regardless of where the launcher is executed from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Optional: suppress Gradio analytics noise ──────────────────────
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GRADIO_PORT = 7860
GRADIO_URL = f"http://127.0.0.1:{GRADIO_PORT}"

# ── Global reference so the webview window can be closed from threads
_window: any = None

# ---------------------------------------------------------------------------
# Setup page — shown when PyTorch is not installed
# ---------------------------------------------------------------------------

_SETUP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Starting Whisper Fine-Tune GUI\u2026</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; color: #333;
  }
  .card {
    background: #fff; border-radius: 16px; padding: 40px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.12);
    max-width: 520px; width: 90%; text-align: center;
  }
  h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
  p { color: #666; line-height: 1.6; margin-bottom: 1.5rem; }
  .status { font-size: 0.9rem; color: #888; margin-top: 1rem; }
  .detail { font-size: 0.8rem; color: #aaa; margin-top: 0.5rem; }
  .spinner {
    display: inline-block; width: 40px; height: 40px;
    border: 4px solid #e0e0e0; border-top-color: #4a90d9;
    border-radius: 50%; animation: spin 0.8s linear infinite;
    margin-bottom: 1rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  button {
    background: #4a90d9; color: #fff; border: none;
    padding: 12px 32px; border-radius: 8px; font-size: 1rem;
    cursor: pointer; transition: background 0.2s;
  }
  button:hover { background: #357abd; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  .hidden { display: none !important; }
  .success { color: #2e7d32; }
  .error { color: #c62828; }
</style>
</head>
<body>
<div class="card">
  <h1>\U0001f399 Whisper Fine-Tune GUI</h1>
  <div class="spinner" id="spinner"></div>
  <div id="status" class="status">Checking dependencies\u2026</div>
  <div id="detail" class="detail"></div>
  <button id="installBtn" class="hidden">Install PyTorch</button>
</div>
<script>
  let pywebviewReady = false;
  let checkInterval = null;

  window.addEventListener('pywebviewready', function() {
    pywebviewReady = true;
    checkDependencies();
  });

  function updateStatus(message, className) {
    document.getElementById('spinner').style.display = (className === 'success' || className === 'error') ? 'none' : 'inline-block';
    document.getElementById('status').textContent = message;
    document.getElementById('status').className = 'status' + (className ? ' ' + className : '');
  }

  function setDetail(message) {
    document.getElementById('detail').textContent = message;
  }

  function showInstallButton() {
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('status').textContent = 'PyTorch is not installed. Click below to install it.';
    document.getElementById('installBtn').classList.remove('hidden');
  }

  function showProgress(message) {
    document.getElementById('spinner').style.display = 'inline-block';
    document.getElementById('status').textContent = message;
    document.getElementById('installBtn').classList.add('hidden');
  }

  function reloadApp() {
    window.location.href = 'http://127.0.0.1:7860';
  }

  function checkDependencies() {
    if (!pywebviewReady) return;

    pywebview.api.check_dependencies().then(function(deps) {
      if (deps.gradio_ready) {
        // Gradio server is already up — redirect immediately
        updateStatus('Starting app\u2026', 'success');
        setDetail('Gradio server is ready.');
        setTimeout(reloadApp, 500);
        return;
      }

      if (deps.gradio_error) {
        updateStatus('Failed to start app: ' + deps.gradio_error, 'error');
        setDetail('Check the console for details.');
        return;
      }

      if (deps.torch) {
        // Torch is installed but Gradio isn't ready yet — poll
        updateStatus('Starting server\u2026', '');
        setDetail('PyTorch detected, waiting for Gradio server to start…');
        checkInterval = setInterval(function() {
          pywebview.api.check_dependencies().then(function(d) {
            if (d.gradio_ready) {
              clearInterval(checkInterval);
              updateStatus('Starting app\u2026', 'success');
              setTimeout(reloadApp, 500);
            } else if (d.gradio_error) {
              clearInterval(checkInterval);
              updateStatus('Failed to start app: ' + d.gradio_error, 'error');
            }
          });
        }, 1000);
      } else {
        // Torch is missing — show install button
        updateStatus('PyTorch is required but not found.', '');
        setDetail('Click the button below to install it (CPU-only, macOS).');
        document.getElementById('installBtn').classList.remove('hidden');
      }
    });
  }

  document.getElementById('installBtn').addEventListener('click', function() {
    showProgress('Installing PyTorch (this may take a few minutes)\u2026');
    setDetail('Downloading from https://download.pytorch.org/whl/cpu …');
    if (pywebviewReady) {
      pywebview.api.install_torch().then(function(result) {
        if (result.success) {
          updateStatus('PyTorch installed successfully! Starting app\u2026', 'success');
          setDetail('');
          setTimeout(reloadApp, 1500);
        } else {
          updateStatus('Installation failed: ' + result.message, 'error');
          setDetail('Check your internet connection and try again.');
          document.getElementById('installBtn').textContent = 'Try Again';
          document.getElementById('installBtn').classList.remove('hidden');
        }
      });
    }
  });
</script>
</body>
</html>
"""


def _make_setup_page_html() -> str:
    """Return the setup-page HTML rendered with current formatting."""
    return _SETUP_HTML


# ---------------------------------------------------------------------------
# PyTorch detection & installation helpers
# ---------------------------------------------------------------------------

def _torch_is_available(timeout: int = 10) -> bool:
    """Return True if ``torch`` can be imported successfully within *timeout* seconds.

    Uses a thread pool so a broken/partial PyTorch install on macOS can't
    hang the main thread indefinitely.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    def _try_import():
        import torch  # noqa: F401
        return True

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_try_import)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.warning(
                "torch import timed out after %ds — treating as unavailable",
                timeout,
            )
            return False
        except ImportError:
            return False


def _install_torch() -> dict:
    """
    Install PyTorch via pip (CPU-only on macOS). Called from the JS bridge
    when the user clicks the install button on the setup page.

    Returns ``{"success": True}`` or ``{"success": False, "message": "..."}``.
    """
    import subprocess

    logger.info("User requested PyTorch installation…")

    # macOS does not have CUDA, so CPU-only is the right choice.
    cmd = [
        sys.executable, "-m", "pip", "install", "torch",
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes should be plenty
        )
        if proc.returncode != 0:
            msg = proc.stderr.strip() or "pip install failed"
            logger.error("PyTorch install failed: %s", msg)
            return {"success": False, "message": msg}

        logger.info("PyTorch installed successfully")
        return {"success": True, "message": ""}

    except subprocess.TimeoutExpired:
        logger.error("PyTorch install timed out")
        return {"success": False, "message": "Installation timed out (10 min)"}
    except Exception as e:
        logger.exception("PyTorch install error")
        return {"success": False, "message": str(e)}


# ---------------------------------------------------------------------------
# Gradio server management
# ---------------------------------------------------------------------------

_gradio_server_started = threading.Event()
_gradio_server_failed = threading.Event()
_gradio_server_error: Optional[str] = None
_gradio_app_instance = None


def _start_gradio_server() -> None:
    """Build and launch the Gradio server in a background thread."""
    global _gradio_app_instance, _gradio_server_error

    try:
        from gui.app import build_app

        app = build_app()
        _gradio_app_instance = app

        logger.info("Gradio server starting on %s", GRADIO_URL)
        _gradio_server_started.set()

        app.launch(
            server_port=GRADIO_PORT,
            share=False,
            debug=False,
            show_error=True,
            prevent_thread_lock=True,  # Don't block — let pywebview run
            inbrowser=False,           # Don't open a browser tab
        )
    except Exception as e:
        _gradio_server_error = str(e)
        _gradio_server_failed.set()
        logger.exception("Gradio server failed to start")


# ---------------------------------------------------------------------------
# JS API — exposed to the pywebview setup page
# ---------------------------------------------------------------------------

class _Api:
    """JavaScript API exposed to the webview via ``pywebview.api.*``."""

    @staticmethod
    def install_torch() -> dict:
        """Called from the setup page install button."""
        return _install_torch()

    @staticmethod
    def check_dependencies() -> dict:
        """
        Check which key packages are available.

        Called by the setup page JS as soon as ``pywebviewready`` fires so the
        UI can dynamically show the correct state instead of static text.
        """
        result: dict = {
            "torch": False,
            "tensorboard": False,
            "gradio_ready": False,
            "gradio_error": None,
        }

        try:
            import torch  # noqa: F401
            result["torch"] = True
        except ImportError:
            pass

        try:
            import tensorboard  # noqa: F401
            result["tensorboard"] = True
        except ImportError:
            pass

        result["gradio_ready"] = _gradio_server_started.is_set()
        if _gradio_server_failed.is_set():
            result["gradio_error"] = _gradio_server_error

        return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    global _window

    # Start the Gradio server on a daemon thread immediately
    gradio_thread = threading.Thread(target=_start_gradio_server, daemon=True)
    gradio_thread.start()

    # Always show the setup page first — the JS bridge will check
    # dependencies dynamically and redirect to Gradio when ready.
    # This eliminates the race between the Gradio thread and window
    # creation, and avoids blocking the main thread on torch detection.
    logger.info("Showing setup page with dynamic dependency checking")
    setup_html = _make_setup_page_html()

    _window = webview.create_window(
        title="Whisper Fine-Tune GUI",
        html=setup_html,
        js_api=_Api(),
        width=1200,
        height=800,
        resizable=True,
        min_size=(600, 400),
        text_select=False,
        zoomable=True,
    )

    webview.start(
        gui="webkit",   # Native macOS WebKit — no Qt needed
        private_mode=False,
        debug=False,
    )

    logger.info("Webview closed — shutting down.")


if __name__ == "__main__":
    import webview
    main()
