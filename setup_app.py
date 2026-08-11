"""
py2app setup script — builds the Whisper Fine-Tune GUI macOS .app bundle.

Usage:
    python setup_app.py py2app

This produces ``dist/Whisper Fine-Tune GUI.app``.

PyTorch is excluded from the bundle to keep it lightweight (~200–300 MB).
On first launch the app will detect the missing dependency and offer to
install it automatically.
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
    # Keep the bundle small — exclude heavy ML frameworks that the user
    # is expected to install separately (or via the in-app installer).
    "excludes": [
        "torch",
        "torchvision",
        "torchaudio",
        "torch.distributed",
        "torch.cuda",
        "caffe2",
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
