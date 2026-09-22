import time

class SentenceBuilder:
    def __init__(self, hold_duration_frames: int = 15, confidence_threshold: float = 0.6,
                 dynamic_confidence_threshold: float = 0.35):
        """
        Manages the accumulated sentence text with a hold-to-confirm mechanism.
        
        Args:
            hold_duration_frames: Consecutive frames a prediction must be held to commit (STATIC mode only).
            confidence_threshold: Minimum confidence to consider a STATIC prediction valid for holding.
            dynamic_confidence_threshold: Minimum confidence for DYNAMIC word commits. This is
                intentionally lower than the static threshold and matches the stabilizer's own
                dynamic threshold — dynamic gestures are inherently less confident single-shot
                predictions, not continuously-held poses, so gating them at the same strict
                static threshold silently blocks otherwise-valid word commits.
        """
        self.buffer = ""
        self.hold_duration_frames = hold_duration_frames
        self.confidence_threshold = confidence_threshold
        self.dynamic_confidence_threshold = dynamic_confidence_threshold
        
        self.current_held_prediction = None
        self.held_frames_count = 0
        
        # Reserved gesture commands (using existing dynamic labels as triggers)
        # Documented choice:
        # "more" -> Space
        # "no" -> Delete/Backspace
        # "stop" -> Clear buffer
        self.cmd_space = "more"
        self.cmd_delete = "no"
        self.cmd_clear = "stop"
        
        self.last_committed_time = 0
        self.cooldown_seconds = 1.5 # cooldown between committing the same word repeatedly

    def process_prediction(self, mode: str, label: str, confidence: float) -> str:
        """
        Processes a single smoothed prediction. If the prediction is held long enough
        above the confidence threshold, it is committed to the sentence buffer.

        Note: STATIC letters are continuously-held poses, so they require being
        seen for several consecutive frames before committing (hold_duration_frames).
        DYNAMIC words are one-shot: the classifier only fires briefly, right when
        the gesture sequence buffer completes, and then resets. Requiring the same
        multi-frame hold for DYNAMIC mode means the word is almost always gone
        before it can accumulate enough held frames, so it silently never commits.
        DYNAMIC predictions are therefore committed as soon as they appear (still
        subject to the cooldown below, to avoid the same word firing twice in a row).

        Returns:
            The newly committed word/char, or None if nothing was committed this frame.
        """
        committed_text = None
        required_hold_frames = 1 if mode == "DYNAMIC" else self.hold_duration_frames
        required_confidence = self.dynamic_confidence_threshold if mode == "DYNAMIC" else self.confidence_threshold

        if label and confidence >= required_confidence:
            if label == self.current_held_prediction:
                self.held_frames_count += 1
            else:
                self.current_held_prediction = label
                self.held_frames_count = 1

            if self.held_frames_count >= required_hold_frames:
                # Check cooldown to prevent rapid-fire repeating
                if time.time() - self.last_committed_time > self.cooldown_seconds:
                    committed_text = self._commit_prediction(mode, label)
                    self.last_committed_time = time.time()
                    self.held_frames_count = 0 # reset after commit
        else:
            self.current_held_prediction = None
            self.held_frames_count = 0

        return committed_text

    def _commit_prediction(self, mode: str, label: str) -> str:
        # Handle commands first
        if label == self.cmd_space:
            self.buffer += " "
            return " "
        elif label == self.cmd_delete:
            if len(self.buffer) > 0:
                # If trailing space, remove it first
                if self.buffer.endswith(" "):
                    self.buffer = self.buffer[:-1]
                
                # Delete last word if dynamic, or last char if static
                if mode == "DYNAMIC":
                    parts = self.buffer.rstrip().split(" ")
                    self.buffer = " ".join(parts[:-1]) + (" " if len(parts) > 1 else "")
                else:
                    self.buffer = self.buffer[:-1]
            return "[DELETE]"
        elif label == self.cmd_clear:
            self.buffer = ""
            return "[CLEAR]"
            
        # Handle normal appending
        if mode == "STATIC":
            self.buffer += label
        else: # DYNAMIC
            # Automatically add space before new word if buffer isn't empty and doesn't end with space
            if self.buffer and not self.buffer.endswith(" "):
                self.buffer += " "
            self.buffer += label
            self.buffer += " "
            
        return label

    def manual_space(self):
        self.buffer += " "

    def manual_delete(self):
        if len(self.buffer) > 0:
            self.buffer = self.buffer[:-1]

    def manual_clear(self):
        self.buffer = ""

    def get_sentence(self):
        return self.buffer.strip()
