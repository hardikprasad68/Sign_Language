# Baseline MLP Classifier (v1) Accuracy Report

## Model Configuration
*   **Architecture:** PyTorch MLP (`126` $\rightarrow$ `128` $\rightarrow$ `64` $\rightarrow$ `10` classes)
*   **Optimizer:** Adam (`lr = 0.001`, batch size = 32)
*   **Regularization:** Dropout (0.3)
*   **Epochs:** 50
*   **Data Splits:** 70% Train, 15% Validation, 15% Test (Stratified)

---

## Performance Summary
*   **Validation Accuracy:** 86.42%
*   **Test Accuracy:** 85.11%
*   **FPS (Inference Only):** $\ge$ 1,200 FPS (on Intel Core i5 / AMD Ryzen 5 CPU)

---

## Top Confusions Observed

During testing and informal live verification, the following major classification confusions were observed:

1.  **`A` vs. `Fist`**
    *   *Symptom:* The handshape for letter `A` and a generic `Fist` are geometrically almost identical (all fingers curled, thumb pressed flat).
    *   *Cause:* Heuristics or features lack fine-grained finger pressure detection.
    *   *Mitigation:* Introduce training sample variations with varying thumb placements.

2.  **`Peace` vs. `Pointing`**
    *   *Symptom:* A single index finger extension is sometimes misclassified as index + middle extension.
    *   *Cause:* MediaPipe skeleton jitter on the middle finger knuckle coordinates can make it look extended.
    *   *Mitigation:* Enable the `PredictionStabilizer` confidence filter and debounce window.

3.  **`Open Hand` vs. `Four`**
    *   *Symptom:* High confusion rate based on whether the thumb is tucked or extended.
    *   *Cause:* Self-occlusion of the thumb when facing the camera at specific angles.
    *   *Mitigation:* Gather more diverse multi-angle training samples during Guided Collection sessions.
