# Shared Error Log: Live Sign Classification Confusions

This collaborative document logs live misclassifications observed during team user testing. Developers should append new entries here to guide dataset enhancements (Phase 1) and model optimization (Phase 2).

---

| Date | Tester | Sign Performed | Predicted Sign | Confidence | Lighting / Camera Angle | Suspected Root Cause & Action Item |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-08-11 | P6 | `Peace` | `Pointing` | 0.82 | Overhead light, front view | Middle finger coordinate jitter. **Action:** Apply the `PredictionStabilizer` to filter transient frame noise. |
| 2026-08-11 | P3 | `A` | `Fist` | 0.74 | Low side-lighting, 45 deg | Silhouette occlusion of the thumb. **Action:** Collect more 45-degree angle samples in guided mode. |
| 2026-08-11 | P1 | `Four` | `Open Hand` | 0.88 | Bright background backlight | Backlight glare reduces contrast. MediaPipe incorrectly predicted thumb extension. **Action:** Record backlit calibration frames. |
| 2026-08-11 | P2 | `Thumbs Up` | `Fist` | 0.65 | Extreme high angle | Wrist rotation makes thumb look curled. **Action:** Train MLP with randomized rotation augmentation. |
