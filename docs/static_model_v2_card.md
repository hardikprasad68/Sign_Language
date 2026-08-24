# Model Card: Static Classifier Model v2

> **Phase 2 Deliverable — Track A**  
> **Model Name:** StaticModelV2  
> **Target Task:** Real-time static ASL alphabet gesture recognition (24 letters `A`–`Y`, excluding `J` and `Z` which require motion)  
> **Weights Path:** `models/mlp_v2.pt`  

---

## 1. Model Overview

*   **Architecture Type:** Deep Residual Multi-Layer Perceptron (`ResidualBlock` with LayerNorm & Dropout)
*   **Input Features:** 126 floats (2 hands $\times$ 21 MediaPipe landmarks $\times$ 3 normalized coordinates $(x, y, z)$)
*   **Output Classes:** 24 classes (`A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`)
*   **Number of Parameters:** ~185,000 trainable parameters
*   **Framework:** PyTorch

---

## 2. Dataset Breakdown & Preprocessing

*   **Dataset Source:** Combined Kaggle public ASL Alphabet dataset + self-recorded multi-signer webcam samples.
*   **Total Dataset Size:** 8,400 samples (350 balanced samples per gesture class).
*   **Splits:** 70% Train (5,880 samples), 15% Validation (1,260 samples), 15% Test (1,260 samples).
*   **Normalization Strategy:** Wrist-origin centering (subtracting landmark 0) and middle finger MCP distance scaling (dividing by distance to landmark 9).
*   **Data Augmentation:** 3D random rotation ($\pm 18^\circ$), Gaussian landmark jitter ($\sigma = 0.02$), scale jittering ($0.85\times$ to $1.15\times$), horizontal mirroring ($50\%$ probability).

---

## 3. Training & Hyperparameters

*   **Optimizer:** AdamW (`lr = 0.001`, `weight_decay = 1e-4`)
*   **Learning Rate Scheduler:** `ReduceLROnPlateau` (factor = 0.5, patience = 5)
*   **Loss Function:** `nn.CrossEntropyLoss()`
*   **Batch Size:** 32
*   **Epochs Trained:** 60 epochs
*   **Regularization:** LayerNorm after linear layers + Dropout ($p = 0.25$)

---

## 4. Benchmark Performance

| Evaluation Metric | Baseline Model v1 | **Tuned Model v2** | Target Threshold |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | 86.42% | **94.85%** | >90.00% |
| **Test Set Accuracy** | 85.11% | **94.12%** | >90.00% |
| **Macro F1-Score** | 0.8490 | **0.9405** | >0.9000 |
| **Inference Latency** | ~0.8 ms | **~0.42 ms** | <5.0 ms |

---

## 5. Error Analysis & Known Weaknesses (A4 Analysis)

### Top Confused Gesture Pairs Identified:
1.  **`A` vs. `S` (Fist with Thumb variations)**
    *   *Symptom:* `A` (thumb resting on the side of index finger) vs `S` (thumb wrapped across fingers).
    *   *Cause:* Self-occlusion of thumb tip when camera is directly facing the palm.
    *   *Fix Implemented:* Multi-angle sample synthesis and 3D rotation augmentation during training.

2.  **`M` vs. `N` (Tucked fingers)**
    *   *Symptom:* `M` (thumb tucked under 3 fingers) vs `N` (thumb tucked under 2 fingers).
    *   *Cause:* Fine knuckle displacement resolution noise in MediaPipe.
    *   *Fix Implemented:* Added PredictionStabilizer temporal majority voting filter ($K=4$ frames).

3.  **`V` vs. `U` (Peace vs Tightly Closed V)**
    *   *Symptom:* Separation distance between index and middle fingers.
    *   *Cause:* Scale variance at extreme distance from camera.
    *   *Fix Implemented:* Euclidean scale invariance normalization.
