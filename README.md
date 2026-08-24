# Sign Language Translator — Phase 2: Static Model Refinement & Dynamic Word Kickoff

> **Real-time sign language translator using MediaPipe Hands, PyTorch Static Residual MLP (v2), and Dynamic GRU Sequence Classifier (v1).**  
> Immediate camera startup · Dual-mode static/dynamic toggle (`[m]`) · Temporal majority-vote smoothing · Automated benchmarks.

---

## Directory Structure

```
Sign Language/
├── src/
│   ├── data/
│   │   ├── merge_datasets.py        # Processes public images to landmarks and merges
│   │   ├── audit.py                 # Audits static class distribution and imbalance
│   │   └── generate_dynamic_data.py # Synthesizes 30-frame dynamic word trajectory sequences
│   ├── models/
│   │   ├── mlp_static.py            # PyTorch MLP baseline static model definition
│   │   ├── static_model.py          # Track A — PyTorch StaticModelV2 (Residual MLP + LayerNorm)
│   │   └── dynamic_model.py         # Track B — PyTorch DynamicGRU sequence model & predictor
│   ├── inference/
│   │   ├── stability.py             # Temporal majority-vote prediction stabilizer
│   │   └── sequence_buffer.py       # Rolling 30-frame FIFO landmark buffer
│   ├── capture.py                   # Module 1 — Webcam capture with DirectShow fallback
│   ├── landmarks.py                 # Module 2 — MediaPipe landmark extraction (supports 2 hands)
│   ├── preprocessing.py             # Coordinate origin-centering, scaling, 3D augmentation & flex angles
│   ├── data_logger.py               # Module 3 — CSV & dynamic sequence logger
│   ├── data_utils.py                # Stratified train/val/test splits loader
│   ├── train.py                     # Track A — Static Model v2 training & hyperparameter tuning loop
│   ├── train_dynamic.py             # Track B — Dynamic GRU Model v1 training loop
│   └── main.py                      # Track C — Integrated dual-mode execution loop ([m] toggle)
├── scripts/
│   └── benchmark.py                 # Track C4 — Automated accuracy, F1, latency & FPS benchmark
├── data/
│   ├── landmarks.csv                # Merged static landmark coordinate dataset
│   └── dynamic/                     # Saved 30-frame sequence binary files (.npy)
├── docs/
│   ├── static_model_v2_card.md      # Model Card for Static Model v2
│   └── dynamic_model_v1_card.md     # Model Card for Dynamic GRU Model v1
├── results/
│   ├── mlp_v1_baseline.md           # Phase 1 baseline report (86.42% val accuracy)
│   ├── static_v2_metrics.txt        # Track A Static Model v2 test evaluation metrics
│   ├── dynamic_v1_metrics.txt       # Track B Dynamic GRU Model v1 test evaluation metrics
│   └── phase2_benchmark.md          # Track C4 complete automated benchmark report
├── models/
│   ├── hand_landmarker.task         # MediaPipe Hands task file
│   ├── mlp_v2.pt                    # Trained Static Model v2 PyTorch weights
│   ├── dynamic_v1.pt                # Trained Dynamic GRU Model v1 PyTorch weights
│   ├── class_labels.txt             # Static alphabet class labels list (24 letters)
│   └── dynamic_labels.txt           # Dynamic word vocabulary list (15 words)
└── requirements.txt                 # Python dependencies configuration
```

---

## Installation & Setup

1. Activate virtual environment:
```powershell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

---

## 1. Track A — Static Model v2 (Alphabet `A`–`Y`)

### Training Static Model v2
Train the upgraded `StaticModelV2` with 3D augmentation and residual blocks:
```powershell
python -u src/train.py --csv data/landmarks.csv --epochs 30 --augment --model-out models/mlp_v2.pt
```
*   **Target Accuracy:** >90.00% validation & test accuracy (Achieved: **94.37% Val / 94.52% Test**).
*   Output artifacts saved to `models/mlp_v2.pt` and `results/static_v2_metrics.txt`.

---

## 2. Track B — Dynamic Word Model v1 (15 Words)

### Generate / Record Dynamic Sequences
Generate synthetic 30-frame sequence dataset for the 15 dynamic words (`hello`, `thanks`, `yes`, `no`, `please`, `help`, `sorry`, `name`, `more`, `stop`, `love`, `want`, `eat`, `drink`, `friend`):
```powershell
python -u src/data/generate_dynamic_data.py --output data/dynamic --samples 50
```

To record live sequences directly via webcam, launch `src/data_logger.py` and press `[d]` to start/stop dynamic sequence recording.

### Train Dynamic GRU Model v1
Train the 2-layer Bidirectional GRU sequence classifier:
```powershell
python -u src/train_dynamic.py --data-dir data/dynamic --epochs 30
```
*   Output artifacts saved to `models/dynamic_v1.pt` and `results/dynamic_v1_metrics.txt`.

---

## 3. Track C — Integrated Live Dual-Mode Demo (`src/main.py`)

Launch the combined application supporting real-time webcam inference and mode toggling:
```powershell
python src/main.py
```

### Keypress Controls inside Application:
*   **`[m]`** : **Toggle Mode** between `Static (Alphabet)` and `Dynamic (Words)` without restarting.
*   **`[s]`** : Save current landmark frame to `data/landmarks.csv`.
*   **`[l]`** : Cycle active static target label.
*   **`[w]`** : Cycle active dynamic word target label.
*   **`[q]`** : Cleanly exit application window.

---

## 4. Track C4 — Automated Benchmarking (`scripts/benchmark.py`)

Run automated held-out test evaluation, inference latency benchmarking, and pipeline FPS measurement:
```powershell
python scripts/benchmark.py
```
*   Generates full benchmark report at `results/phase2_benchmark.md`.

---

## 5. Phase 3 — Full Alpha Application (`src/ui_app.py`)

Turned the real-time pipeline into a usable application with a PyQt6 GUI, text accumulation, and text-to-speech.

### Launch the App
```powershell
python src/ui_app.py
```

### Features & Gesture Commands
- **Accumulated Sentence**: Build full sentences using the hold-to-confirm mechanism. Wait for the "Hold" progress to reach 100% to commit a prediction.
- **Gesture Commands** (When in DYNAMIC Mode):
  - **Space**: Sign the word `"more"`
  - **Delete/Backspace**: Sign the word `"no"`
  - **Clear Buffer**: Sign the word `"stop"`
- **Text-to-Speech**: Click "Speak Sentence" or toggle "Auto-Speak" to read text aloud using `pyttsx3`.
- **Robustness**: Application survives camera disconnects and handles 'no hand' frames gracefully.

**Known Limitations for Alpha**:
- Webcams with poor lighting or high latency may drop frames or struggle with DYNAMIC sequences.
- Pyttsx3 TTS is synchronous internally and may cause a micro-stutter if auto-speak is enabled, although threaded. 
- Fast signers might need to lower the "Hold Duration" slider in settings.
