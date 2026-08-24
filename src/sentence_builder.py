import time

class SentenceBuilder:
    def __init__(self, hold_duration_frames: int = 15, confidence_threshold: float = 0.6):
        """
        Manages the accumulated sentence text with a hold-to-confirm mechanism.
        
        Args:
            hold_duration_frames: Consecutive frames a prediction must be held to commit.
            confidence_threshold: Minimum confidence to consider a prediction valid for holding.
        """
        self.buffer = ""
        self.hold_duration_frames = hold_duration_frames
        self.confidence_threshold = confidence_threshold
        
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
        
        Returns:
            The newly committed word/char, or None if nothing was committed this frame.
        """
        committed_text = None
        
        if label and confidence >= self.confidence_threshold:
            if label == self.current_held_prediction:
                self.held_frames_count += 1
            else:
                self.current_held_prediction = label
                self.held_frames_count = 1
                
            if self.held_frames_count >= self.hold_duration_frames:
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
