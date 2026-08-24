"""Shared ASL alphabet label definitions used across the pipeline."""

from pathlib import Path

# Static ASL letters (J and Z require motion and are excluded)
ASL_STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

_DEFAULT_LABELS_PATH = Path(__file__).resolve().parent.parent / "models" / "class_labels.txt"


def load_class_labels(labels_path: str | None = None) -> list[str]:
    """Load class labels from file, falling back to the built-in ASL list."""
    path = Path(labels_path) if labels_path else _DEFAULT_LABELS_PATH
    if path.exists():
        with open(path, "r") as f:
            labels = [line.strip() for line in f.read().splitlines() if line.strip()]
        if labels:
            return labels
    return ASL_STATIC_LETTERS.copy()


_DEFAULT_DYNAMIC_LABELS_PATH = Path(__file__).resolve().parent.parent / "models" / "dynamic_labels.txt"
DEFAULT_DYNAMIC_VOCAB = ["hello", "thanks", "yes", "no", "please", "help", "sorry", "name", "more", "stop", "love", "want", "eat", "drink", "friend"]

def load_dynamic_labels(labels_path: str | None = None) -> list[str]:
    """Load dynamic word labels from file, falling back to default vocabulary."""
    path = Path(labels_path) if labels_path else _DEFAULT_DYNAMIC_LABELS_PATH
    if path.exists():
        with open(path, "r") as f:
            labels = [line.strip() for line in f.read().splitlines() if line.strip()]
        if labels:
            return labels
    return DEFAULT_DYNAMIC_VOCAB.copy()

