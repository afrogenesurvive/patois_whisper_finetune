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

import asyncio
import logging
import os
import shutil
import socket
import subprocess
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

# ── File logging ──────────────────────────────────────────────────
# Set up the file log at IMPORT time (not inside main()) so that even
# import-time errors and early crashes are captured to disk.  When the
# .app is launched from Finder/Dock there is no terminal, so we also
# redirect stderr/stdout into the same log file — that way uncaught
# tracebacks and print() output are preserved for diagnosis.
_LOG_DIR = Path.home() / ".whisper-gui"
_LOG_FILE: Optional[Path] = None


def _setup_file_logging() -> Optional[Path]:
    """Install a FileHandler on the root logger; rotate the previous log.

    Returns the log file path, or None if it could not be created.
    """
    global _LOG_FILE
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Keep the previous run's log for comparison before truncating.
        prev = _LOG_DIR / "app.log.prev"
        cur = _LOG_DIR / "app.log"
        if cur.exists():
            try:
                shutil.copy2(cur, prev)
            except OSError:
                pass

        handler = logging.FileHandler(str(cur), mode="w", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logging.getLogger().addHandler(handler)

        # When launched from Finder/Dock (no TTY), capture raw stdout/stderr
        # so uncaught exceptions and C-level tracebacks end up in the log.
        if sys.stderr is None or not getattr(
            sys.stderr, "isatty", lambda: True
        )():
            sys.stderr = open(str(cur), "a", encoding="utf-8")
        if sys.stdout is None or not getattr(
            sys.stdout, "isatty", lambda: True
        )():
            sys.stdout = open(str(cur), "a", encoding="utf-8")

        _LOG_FILE = cur
        logger.info("Logging to %s", cur)
    except Exception:  # pragma: no cover - never let logging break startup
        logger.exception("Could not set up file logging")
        _LOG_FILE = None
    return _LOG_FILE


_setup_file_logging()

# ── Gradio server ─────────────────────────────────────────────────
# Default (preferred) port.  A busy port used to crash the app with
# "Cannot find empty port in range: 7860-7860"; we now fall back to the
# next free port automatically (see ``_pick_port`` / ``main()``).
GRADIO_PORT = 7860
_gradio_port: int = GRADIO_PORT
_gradio_url: str = f"http://127.0.0.1:{_gradio_port}"


def _pick_port(preferred: int = GRADIO_PORT, max_attempts: int = 20) -> int:
    """Return *preferred* if free, otherwise the next free port >= preferred.

    Used at startup so the Gradio server always gets a port it can bind,
    instead of failing when 7860 is already occupied.
    """
    for port in range(preferred, preferred + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        "No free port found in range "
        f"{preferred}..{preferred + max_attempts - 1}"
    )

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
  .quit-btn {
    position: fixed; top: 12px; left: 12px; z-index: 1000;
    background: rgba(60,60,60,0.85); color: #fff;
    border: none; border-radius: 8px;
    padding: 8px 14px; font-size: 0.85rem; font-weight: 600;
    cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.25);
  }
  .quit-btn:hover { background: #c0392b; }
  .quit-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.45);
    display: flex; align-items: center; justify-content: center;
    z-index: 5000;
  }
  .quit-dialog {
    background: #fff; border-radius: 12px;
    padding: 24px 28px; box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    max-width: 360px; text-align: center;
  }
  .quit-dialog h3 { margin: 0 0 8px; font-size: 1.1rem; }
  .quit-dialog p { color: #666; margin: 0 0 18px; font-size: 0.9rem; }
  .quit-actions { display: flex; gap: 10px; justify-content: center; }
  .quit-actions button {
    border: none; border-radius: 8px;
    padding: 8px 18px; font-size: 0.85rem; font-weight: 600; cursor: pointer;
  }
  .quit-cancel { background: #e2e2e2; color: #333; }
  .quit-cancel:hover { background: #d0d0d0; }
  .quit-confirm { background: #c0392b; color: #fff; }
  .quit-confirm:hover { background: #a93226; }
  .success { color: #2e7d32; }
  .error { color: #c62828; }
  .console-wrap { margin-top: 1rem; text-align: left; }
  .console-wrap summary { cursor: pointer; font-size: 0.85rem; color: #666; padding: 4px 0; }
  .console-wrap summary:hover { color: #333; }
  .console-output {
    background: #1e1e1e; color: #d4d4d4;
    padding: 12px; border-radius: 8px;
    font-size: 0.75rem; line-height: 1.5;
    max-height: 250px; overflow-y: auto;
    text-align: left; white-space: pre-wrap;
    word-break: break-all;
    font-family: 'SF Mono', Monaco, Menlo, 'Courier New', monospace;
  }
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
<button id="quitBtn" class="quit-btn" title="Quit the application">✕ Quit App</button>
<div id="quitOverlay" class="quit-overlay" style="display:none;">
  <div class="quit-dialog">
    <h3>Quit Whisper Fine-Tune GUI?</h3>
    <p>Any in-progress work will be lost.</p>
    <div class="quit-actions">
      <button id="quitCancel" class="quit-cancel">Cancel</button>
      <button id="quitConfirm" class="quit-confirm">Quit</button>
    </div>
  </div>
</div>
<details class="console-wrap">
  <summary>Console Log</summary>
  <pre class="console-output" id="consoleOutput">Waiting for logs…</pre>
</details>
<script>
  let pywebviewReady = false;
  let checkInterval = null;

  document.getElementById('quitBtn').addEventListener('click', function() {
    document.getElementById('quitOverlay').style.display = 'flex';
  });
  document.getElementById('quitCancel').addEventListener('click', function() {
    document.getElementById('quitOverlay').style.display = 'none';
  });
  document.getElementById('quitConfirm').addEventListener('click', function() {
    if (window.pywebview) { pywebview.api.quit_app(); }
  });

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
    pywebview.api.get_gradio_url().then(function(url) {
      pywebview.api.navigate_to(url);
    });
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

  // ── Log polling (runs independently of dependency checks) ──────
  setInterval(function() {
    if (pywebviewReady) {
      pywebview.api.get_recent_logs().then(function(logs) {
        var el = document.getElementById('consoleOutput');
        el.textContent = logs || '(no logs yet)';
        el.scrollTop = el.scrollHeight;
      });
    }
  }, 1000);
</script>
</body>
</html>
"""


def _make_setup_page_html() -> str:
    """Return the setup-page HTML (the JS fetches the live Gradio URL)."""
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


def _find_python_for_pip() -> Optional[str]:
    """
    Return a working Python interpreter path that has pip access.

    In a py2app bundle ``sys.executable`` points to a non-runnable stub
    inside the .app (e.g. ``.../MacOS/Python3``) which doesn't exist as a
    standalone binary.  This function falls back to the build venv, then
    PATH, to find a real Python with pip.
    """
    # 1. sys.executable if it's an actual file (normal Python, not bundled)
    if sys.executable and os.path.isfile(sys.executable):
        return sys.executable

    _root = Path(__file__).resolve().parent.parent

    # 2. venv that was used to build the .app (if it still exists)
    venv_python = _root / "venv" / "bin" / "python3"
    if venv_python.is_file():
        return str(venv_python)

    # 3. PATH lookup
    which_python = shutil.which("python3")
    if which_python:
        return which_python

    # 4. Last resort — might fail, but we tried
    return "python3"


def _install_torch() -> dict:
    """
    Install PyTorch via pip (CPU-only on macOS). Called from the JS bridge
    when the user clicks the install button on the setup page.

    Returns ``{"success": True}`` or ``{"success": False, "message": "..."}``.
    """
    logger.info("User requested PyTorch installation…")

    python_cmd = _find_python_for_pip()
    logger.info("Using Python interpreter: %s", python_cmd)

    # macOS does not have CUDA, so CPU-only is the right choice.
    cmd = [
        python_cmd, "-m", "pip", "install", "torch",
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


def _wait_for_server(port: int, timeout: float = 10.0) -> None:
    """
    Poll *port* on 127.0.0.1 until a TCP connection is accepted.

    This prevents the JS from navigating to the Gradio URL before the
    server is ready, which would cause infinite "connection errored"
    notifications in the webview.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.2)
    raise RuntimeError(
        f"Gradio server did not start on port {port} within {timeout}s"
    )


def _start_gradio_server() -> None:
    """Build and launch the Gradio server in a background thread.

    The chosen port can be grabbed by another process between the initial
    availability check (in ``main()``) and the actual bind — the heavy
    ``build_app()`` import leaves a window where a race is possible.  On a
    port conflict we re-pick a free port and retry.
    """
    global _gradio_app_instance, _gradio_server_error, _gradio_port, _gradio_url

    # Create an event loop for this background thread.  Python 3.9's
    # ``asyncio.Lock()`` constructor calls ``get_event_loop()``, so
    # Gradio's ``safe_get_lock()`` (and anything else that needs an
    # asyncio context) needs a loop to be available before ``build_app``
    # constructs the ``Queue`` object.
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except RuntimeError:
        pass  # Should not happen in a fresh thread, but be defensive.

    from gui.app import build_app

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            app = build_app()
            _gradio_app_instance = app

            logger.info(
                "Gradio server starting (attempt %d/%d) on %s",
                attempt, max_attempts, _gradio_url,
            )

            app.launch(
                server_port=_gradio_port,
                share=False,
                debug=False,
                show_error=False,          # Suppress connection-error notifications
                quiet=True,                # Suppress Gradio's internal print noise
                prevent_thread_lock=True,  # Don't block — let pywebview run
                inbrowser=False,           # Don't open a browser tab
            )

            # Only signal ready once the server is actually accepting connections
            _wait_for_server(_gradio_port, timeout=10.0)
            logger.info("Gradio server is ready on %s", _gradio_url)
            _gradio_server_started.set()
            return

        except OSError as e:
            # Port conflict — log who holds it, then pick the next free port.
            if "port" in str(e).lower() and attempt < max_attempts:
                try:
                    holder = subprocess.run(
                        ["lsof", "-nP", "-iTCP:%d" % _gradio_port],
                        capture_output=True, text=True, timeout=5,
                    ).stdout.strip()
                except Exception:
                    holder = ""
                if holder:
                    logger.warning(
                        "Port %d is held by:\n%s", _gradio_port, holder
                    )
                new_port = _pick_port(_gradio_port + 1)
                logger.warning(
                    "Port %d was busy (%s) — retrying on %d",
                    _gradio_port, e, new_port,
                )
                _gradio_port = new_port
                _gradio_url = f"http://127.0.0.1:{new_port}"
                continue

            _gradio_server_error = str(e)
            _gradio_server_failed.set()
            logger.exception("Gradio server failed to start")
            return

        except Exception as e:
            _gradio_server_error = str(e)
            _gradio_server_failed.set()
            logger.exception("Gradio server failed to start")
            return


# ---------------------------------------------------------------------------
# JS API — exposed to the pywebview setup page
# ---------------------------------------------------------------------------

class _Api:
    """JavaScript API exposed to the webview via ``pywebview.api.*``."""

    @staticmethod
    def navigate_to(url: str) -> None:
        """Navigate the webview window to *url* (called from JS)."""
        global _window
        if _window:
            _window.load_url(url)

    @staticmethod
    def get_gradio_url() -> str:
        """Return the current Gradio URL (port can change after a retry)."""
        return _gradio_url

    @staticmethod
    def get_recent_logs(n: int = 50) -> str:
        """Return the last *n* log lines as a single string (called from JS)."""
        from gui.utils import log_buffer
        return "\n".join(log_buffer.get_logs(n))

    @staticmethod
    def install_torch() -> dict:
        """Called from the setup page install button."""
        return _install_torch()

    @staticmethod
    def quit_app() -> None:
        """Quit the application (called by the Quit App button in the UI)."""
        global _window
        logger.info("User confirmed app quit…")
        w = _window
        if not w:
            return
        # Defer the destroy so the JS bridge can send its response before the
        # window is torn down. Calling destroy() synchronously from a JS API
        # call hangs the app on macOS (the bridge then tries to evaluate JS
        # against a destroyed webview).
        def _do_quit():
            time.sleep(0.2)
            try:
                w.destroy()
            except Exception:
                logger.exception("Failed to destroy window")
        threading.Thread(target=_do_quit, daemon=True).start()

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
    global _window, _gradio_port, _gradio_url

    # Pick a free port up-front so the setup page HTML can embed the real URL.
    _gradio_port = _pick_port()
    _gradio_url = f"http://127.0.0.1:{_gradio_port}"
    logger.info("Gradio will use port %d (%s)", _gradio_port, _gradio_url)

    # Install the shared log buffer (Option C: in-app log viewing)
    from gui.utils import log_buffer
    logging.getLogger().addHandler(log_buffer)

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
        text_select=True,   # Allow selecting/copying text in the UI
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
