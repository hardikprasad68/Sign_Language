import threading
import pyttsx3

class SpeechEngine:
    def __init__(self, auto_speak: bool = False):
        self.auto_speak = auto_speak
        self.engine = None
        self.enabled = False
        
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150) # Slower, more readable pace
            self.enabled = True
        except Exception as e:
            print(f"[WARNING] TTS initialization failed: {e}. Speech output will be disabled.")
            
        self.lock = threading.Lock()

    def speak(self, text: str):
        if not self.enabled or not text:
            return
            
        # Run speech in a separate thread so it doesn't block the video loop
        thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        thread.start()
        
    def _speak_thread(self, text: str):
        with self.lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"[WARNING] TTS playback failed: {e}")

    def toggle_auto_speak(self):
        self.auto_speak = not self.auto_speak
        return self.auto_speak
