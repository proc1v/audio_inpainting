import whisper

model = whisper.load_model("base")
result = model.transcribe("data/harvard.wav")
result2 = model.transcribe("data/jackhammer.wav")
print(result["text"])
print(result2["text"])