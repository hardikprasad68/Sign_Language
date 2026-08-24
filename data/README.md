# Hand Landmark Sign Dataset Documentation

This directory contains the landmark coordinate datasets for the Real-Time Sign Language Translator.

## Schema Specifications
Recorded in `landmarks.csv`. Each row contains **129 columns**:
1.  **Columns 0 to 125 (`x0,y0,z0` to `x41,y41,z41`):** Flat coordinate positions of up to two hands (wrist-normalized, scale-normalized, padded with `0.0` for missing hands).
2.  **Column 126 (`label`):** Gesture target sign (e.g. `A`, `Peace`, `Thumbs Up`).
3.  **Column 127 (`timestamp`):** Unix timestamp of sample insertion.
4.  **Column 128 (`session_tag`):** Session categorization tag.

## Dataset Class Distribution (Initial Draft)
*   **Public Dataset:** ASL Alphabet (processed image bounding boxes converted to keypoint lists).
*   **Self-Recorded Dataset:** Camera frame logs captured via `data_logger.py`.

*Note: The combined total count is automatically updated when running `src/data/merge_datasets.py`.*
