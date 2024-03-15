import speech_recognition as sr

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize_local_whisper(self, audio_file):
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
        
        try:
            # Recognize using the whisper model (local)
            text = self.recognizer.recognize_whisper(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return f"Error: {str(e)}"
    
    def recognize_google_api(self, audio_file, language="en-US"):
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
        
        try:
            # Recognize using Google Web Speech API
            text = self.recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return f"Error: {str(e)}"
