import numpy as np
from transformers import pipeline

class SpeechRecognizer:
    def __init__(self, model: str = "openai/whisper-base.en", device: str = "cpu"):
        self.transcriber = pipeline("automatic-speech-recognition", model=model, device=device)
        
    def transcribe(self, audio):
        sr, y = audio
        if y.ndim == 2:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        
        return self.transcriber({"sampling_rate": sr, "raw": y})["text"]
       