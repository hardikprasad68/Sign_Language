"""
setup_check.py — Environment Verification Script
=================================================
Every team member runs this script after setting up their environment.
It checks Python version, installed packages, webcam access, and
import health for all pipeline modules.

Usage:
    python scripts/setup_check.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import sys
import importlib
import platform


def _check(label: str, passed: bool, detail: str = "") -> bool:
    """Print a check result and return whether it passed."""
    icon = "✓" if passed else "✗"
    msg = f"  [{icon}] {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)
    return passed


def main() -> int:
    print("=" * 60)
    print("  Sign Language Translator — Environment Check")
    print("=" * 60)
    print(f"  OS:      {platform.system()} {platform.release()}")
    print(f"  Python:  {sys.version}")
    print("=" * 60)
    print()

    all_passed = True

    # ── 1. Python version ────────────────────────────────────────
    print("[Python Version]")
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= 10
    detail = f"{major}.{minor}" + ("" if ok else " — need 3.10+")
    all_passed &= _check("Python >= 3.10", ok, detail)
    if major == 3 and minor >= 12:
        _check("Python < 3.12 (recommended for MediaPipe)", False,
               "mediapipe may not work on 3.12+; use 3.10 or 3.11")
    print()

    # ── 2. Required packages ─────────────────────────────────────
    print("[Required Packages]")
    required = {
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
    }
    for import_name, pip_name in required.items():
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "unknown")
            all_passed &= _check(f"{pip_name} installed", True, f"version {ver}")
        except ImportError:
            all_passed &= _check(f"{pip_name} installed", False,
                                 f"pip install {pip_name}")
    print()

    # ── 3. Optional packages (for later phases) ──────────────────
    print("[Optional Packages (Phase 1+)]")
    optional = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "torch": "torch (PyTorch)",
        "tensorflow": "tensorflow",
    }
    for import_name, pip_name in optional.items():
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "unknown")
            _check(f"{pip_name}", True, f"version {ver}")
        except ImportError:
            _check(f"{pip_name}", False, "not installed (ok for Phase 0)")
    print()

    # ── 4. Pipeline module imports ────────────────────────────────
    print("[Pipeline Modules]")
    # We need to be able to import from the project root, so add parent dir
    project_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    modules = ["capture", "detection", "classifier", "logger", "main"]
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            all_passed &= _check(f"import {mod_name}", True)
        except Exception as e:
            all_passed &= _check(f"import {mod_name}", False, str(e))
    print()

    # ── 5. Webcam access ──────────────────────────────────────────
    print("[Webcam Access]")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                all_passed &= _check("Webcam readable", True,
                                     f"{w}×{h} frame captured")
            else:
                all_passed &= _check("Webcam readable", False,
                                     "opened but could not read a frame")
            cap.release()
        else:
            all_passed &= _check("Webcam accessible", False,
                                 "cv2.VideoCapture(0) failed — check permissions")
    except Exception as e:
        all_passed &= _check("Webcam check", False, str(e))
    print()

    # ── 6. Classifier self-test ───────────────────────────────────
    print("[Classifier Self-Test]")
    try:
        from classifier import predict

        class _FakeLM:
            def __init__(self, x, y, z=0.0):
                self.x, self.y, self.z = x, y, z

        # Quick smoke test: all fingers extended → should return "Open Hand"
        lms = []
        lms.append(_FakeLM(0.4, 0.9))     # wrist
        lms.append(_FakeLM(0.5, 0.85))    # thumb CMC
        lms.append(_FakeLM(0.55, 0.80))   # thumb MCP
        lms.append(_FakeLM(0.60, 0.76))   # thumb IP
        lms.append(_FakeLM(0.65, 0.72))   # thumb TIP (extended)
        for i in range(4):  # index, middle, ring, pinky (MCP, PIP, DIP, TIP)
            base_x = 0.4 + i * 0.07
            lms.append(_FakeLM(base_x, 0.75))   # MCP
            lms.append(_FakeLM(base_x, 0.65))   # PIP
            lms.append(_FakeLM(base_x, 0.58))   # DIP
            lms.append(_FakeLM(base_x, 0.50))   # TIP (extended — above PIP)

        result = predict(lms)
        all_passed &= _check("predict() returns a label", bool(result),
                             f"got '{result}'")
    except Exception as e:
        all_passed &= _check("Classifier self-test", False, str(e))
    print()

    # ── Summary ───────────────────────────────────────────────────
    print("=" * 60)
    if all_passed:
        print("  ✓ ALL CHECKS PASSED — environment is ready!")
    else:
        print("  ✗ SOME CHECKS FAILED — fix the issues above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
