"""
py2app setup script — builds the Whisper Fine-Tune GUI macOS .app bundle.

Usage:
    python setup_app.py py2app

This produces ``dist/Whisper Fine-Tune GUI.app``.

PyTorch (CPU) is bundled into the app at build time so the app runs
out-of-the-box. This makes the bundle large (~1 GB); the tradeoff is
that the runtime "Install PyTorch" flow is no longer needed.
"""

import sys

# py2app's modulegraph AST visitor can exceed Python's default recursion
# limit (1000) when scanning deeply nested packages like Gradio or
# Transformers.  Bump it to a safe value before setup() is called.
sys.setrecursionlimit(5000)

from setuptools import setup

APP = ["gui/launcher.py"]

# PyObjCTools is a namespace package (no __init__.py), so py2app's modulegraph
# can't resolve it automatically.  Copy it as a data file so pywebview's
# cocoa platform can import PyObjCTools.AppHelper et al.
import os as _os
_PYOBJCTOOLS = _os.path.join(
    _os.path.dirname(__file__),
    "venv/lib/python3.9/site-packages/PyObjCTools",
)
_PYOBJCTOOLS_FILES = sorted(
    _os.path.join(_PYOBJCTOOLS, f)
    for f in _os.listdir(_PYOBJCTOOLS)
    if f.endswith(".py")
)
DATA_FILES = [("lib/python3.9/PyObjCTools", _PYOBJCTOOLS_FILES)]

OPTIONS = {
    "argv_emulation": False,
    # torch is bundled (not excluded) — the app cannot run without it.
    # Only the optional torchvision/torchaudio companions are excluded;
    # they are not installed in the build venv and are not required for
    # CPU-only training/inference.
    "excludes": [
        "torchvision",
        "torchaudio",
    ],
    "packages": [
        "gui",
        "gui.tabs",
        "scripts",
        "webview",
        "gradio",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "librosa",
        "soundfile",
        "jiwer",
        "evaluate",
        "plotly",
        "tensorboard",
        "yaml",
        "tqdm",
        # PyTorch — bundled at build time.  Include its runtime deps so
        # modulegraph doesn't miss any pulled in dynamically.
        "torch",
        "filelock",
        "fsspec",
        "jinja2",
        "mpmath",
        "networkx",
        "sympy",
        "typing_extensions",
        # PyObjC framework packages — required by webview.platforms.cocoa
        "objc",
        "AppKit",
        "Foundation",
        "WebKit",
        "CoreFoundation",
        "Quartz",
    ],
    "includes": [
        "webview.platforms.cocoa",
        # Also ensure all PyObjC .so extensions are captured
        "objc._objc",
        "AppKit._AppKit",
        "Foundation._Foundation",
        "WebKit._WebKit",
        # uvicorn/anyio resolve these via dynamic string imports, so modulegraph
        # misses them — without them the Gradio server fails with errors like
        # "Could not import module 'uvicorn.protocols.http.auto'", "No module
        # named 'uvicorn.lifespan'", or "No module named 'anyio._backends'".
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.loops.uvloop",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.wsproto_impl",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.protocols.websockets.websockets_sansio_impl",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "uvicorn.lifespan.off",
        "uvicorn.__main__",
        "anyio._backends",
        "anyio._backends._asyncio",
    ],
    "plist": {
        "CFBundleName": "Whisper Fine-Tune GUI",
        "CFBundleDisplayName": "Whisper Fine-Tune GUI",
        "CFBundleIdentifier": "com.yourname.whisper-patois-gui",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleExecutable": "Whisper Fine-Tune GUI",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
        "NSHumanReadableCopyright": (
            "Copyright © 2026. All rights reserved."
        ),
    },
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
