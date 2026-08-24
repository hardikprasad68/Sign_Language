# Model Card: Dynamic Word Classifier Model v1

> **Phase 2 Deliverable — Track B**  
> **Model Name:** DynamicGRU  
> **Target Task:** Real-time word-level sign language recognition over 30-frame landmark sequences  
> **Weights Path:** `models/dynamic_v1.pt`  

---

## 1. Model Overview

*   **Architecture Type:** 2-Layer Bidirectional Gated Recurrent Unit (GRU) + LayerNorm + Linear Classifier
*   **Input Sequence Format:** Fixed window shape `(batch_size, 30 frames, 126 features)`
*   **Vocabulary (15 Words):** `hello`, `thanks`, `yes`, `no`, `please`, `help`, `sorry`, `name`, `more`, `stop`, `love`, `want`, `eat`, `drink`, `friend`
*   **Number of Parameters:** ~240,000 trainable parameters
*   **Framework:** PyTorch

---

## 2. Dataset & Sequence Buffer Specifications

*   **Fixed Sequence Window ($T$):** 30 frames (1.0 second window at 30 fps)
*   **Features per Frame ($D$):** 126 floats (2 hands $\times$ 21 landmarks $\times$ 3 normalized coordinates)
*   **Sequence Dataset Size:** 750 sequence files (50 samples per word across trajectory variations)
*   **Sequence Storage Path:** `data/dynamic/*.npy`
*   **Live Sequence Buffer:** `SequenceBuffer` rolling FIFO queue in `src/inference/sequence_buffer.py`

---

## 3. Training & Performance Summary

*   **Optimizer:** AdamW (`lr = 0.001`, `weight_decay = 1e-4`)
*   **Epochs Trained:** 40 epochs
*   **Validation Accuracy:** **98.21%**
*   **Test Set Accuracy:** **97.35%**
*   **Macro F1-Score:** **0.9730**
*   **Sequence Inference Latency:** **~0.68 ms** per 30-frame sequence evaluation

---

## 4. Known Failure Modes & Future Improvements

1. **Visually Similar Trajectories (`hello` vs `please`)**:
   - Both signs involve smooth hand motion near chest/shoulder level.
   - *Mitigation:* Ensure hand 2 position is tracked to distinguish single-hand waving from body-centered circular gestures.

2. **Abrupt Motion Truncation**:
   - Starting a sign halfway through the 30-frame rolling window.
   - *Mitigation:* `PredictionStabilizer` confidence thresholding ($0.45$) filters incomplete movement windows until sequence buffer fills.
