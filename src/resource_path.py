"""
resource_path.py — Resolve project paths correctly whether running from
source (python src/ui_app.py) or as a PyInstaller-frozen .exe.

When PyInstaller builds a onefile executable, bundled data files get
extracted to a temporary folder at runtime (sys._MEIPASS), which has a
different layout than the source tree. Code that computes paths via
`Path(__file__).resolve().parent.parent` breaks in that case. This helper
detects which mode we're in and returns the right root either way.
"""

import sys
from pathlib import Path


def get_project_root() -> Path:
    if getattr(sys, "frozen", False):
        # PyInstaller sets sys._MEIPASS, but where bundled data (via --add-data)
        # actually ends up varies by version: some onedir builds place it
        # directly at _MEIPASS, others place it in an _internal folder next
        # to the .exe. Check every plausible location and use whichever one
        # actually has our bundled "models" folder in it.
        candidates = []
        if hasattr(sys, "_MEIPASS"):
            candidates.append(Path(sys._MEIPASS))
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir)
        candidates.append(exe_dir / "_internal")

        for candidate in candidates:
            if (candidate / "models").exists():
                return candidate

        # Nothing matched — fall back to the first candidate so the caller
        # at least gets a sensible (if possibly wrong) path rather than a crash.
        return candidates[0] if candidates else exe_dir

    # Running from source: this file lives in src/, project root is one level up
    return Path(__file__).resolve().parent.parent
