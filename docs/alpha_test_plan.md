# Phase 3 Alpha Testing Plan & Feedback Template

## Overview
This test is designed for outside users to evaluate the real-time usability, UI clarity, and stability of the Phase 3 Sign Language Translator application.

## Prerequisites
- Install dependencies: `pip install -r requirements.txt`
- Launch the application: `python src/ui_app.py`
- Ensure webcam is connected and unblocked.

## Test Script (Tasks for the Tester)

### Task 1: Basic Initialization & UI
1. Start the application and click "Start Camera".
2. **Observe:** Does the camera feed appear without freezing?
3. **Observe:** Is the "No hand detected" warning visible when you step away, and does it disappear when you show your hand?

### Task 2: Static Fingerspelling (STATIC Mode)
1. Ensure the mode dropdown is set to **STATIC (Alphabet)**.
2. Sign the letter **'H'**, hold it until the hold progress reaches 100%.
    - *Expected:* 'H' appears in the Accumulated Sentence buffer.
3. Sign **'E'**, **'L'**, **'L'**, **'O'**, holding each to confirm.
    - *Expected:* Buffer reads "HELLO".
4. Sign the gesture for **Space** (dynamic word 'more' or click the Space button).
    - *Expected:* A space is added.

### Task 3: Dynamic Words (DYNAMIC Mode)
1. Switch the mode dropdown to **DYNAMIC (Words)**.
2. Sign the word **"hello"** and hold the end pose.
    - *Expected:* "hello " is appended to the buffer.
3. Sign **"friend"**.
    - *Expected:* "friend " is appended.
4. Try using the gesture commands (if you can comfortably sign them) or the UI buttons:
    - Click **Delete**: the last word should be removed.
    - Click **Clear**: the entire buffer should empty.

### Task 4: Text-to-Speech (TTS)
1. Check the "Auto-Speak Confirmed Words" box.
2. Sign a known word/letter. 
    - *Expected:* The system speaks the word as soon as it commits.
3. Type a sentence via gestures, then click **"Speak Sentence"**.
    - *Expected:* The entire accumulated sentence is spoken aloud.

### Task 5: Robustness
1. While the app is running, unplug or disable your webcam.
    - *Expected:* The app should not crash. A warning popup or text should appear.
2. Reconnect the camera and click Start Camera again.
    - *Expected:* The feed should resume normally.

---

## Tester Feedback Template

**Tester Name/ID:** 
**Date:**
**System/OS:**

### 1. Accuracy & Responsiveness
- Was the hold-to-confirm duration (default 15 frames) too fast, too slow, or just right?
  - *Notes:* 
- Did the system frequently double-type letters or words?
  - *Notes:* 

### 2. Usability
- Was it clear when a prediction was about to be committed?
  - *Notes:*
- Were the gesture commands (Space/Delete/Clear) easy to trigger intentionally, and hard to trigger accidentally?
  - *Notes:*

### 3. Stability
- Did the app crash or freeze at any point? If so, what were you doing?
  - *Notes:*

### 4. General Feedback
- What is the most frustrating part of using the app right now?
  - *Notes:*
- Any other suggestions for the UI?
  - *Notes:*
