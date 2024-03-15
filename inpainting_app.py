import gradio as gr
import PIL
import numpy as np
import io
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr


import gradio as gr
from transformers import pipeline
import numpy as np

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    sr, y = audio
    y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]



# Function to process the image and audio inputs
# def process_input(image, audio):
#     # Load the image from input
#     try:
#         image = PIL.Image.open(io.BytesIO(image))
#     except:
#         # make random image if image is not provided
#         image = PIL.Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        
#     #print(*audio)
#     # Convert audio from bytes to AudioSegment
#     audio = AudioSegment.from_file(io.BytesIO(audio[-1]))
    
#     # Play the audio to let user describe what to remove
#     play(audio)
    
#     # Speech recognition
#     recognizer = sr.Recognizer()
#     with io.BytesIO(audio.export(format="wav")) as wav_file:
#         wav_file.seek(0)
#         audio_data = recognizer.record(wav_file)
    
#     try:
#         # Recognize speech from audio
#         transcribed_text = recognizer.recognize_google(audio_data)
#     except sr.UnknownValueError:
#         transcribed_text = "Speech recognition could not understand the audio"
#     except sr.RequestError as e:
#         transcribed_text = f"Error: {str(e)}"
    
#     # Inpainting process (dummy process)
#     # Here, we will just blur the image as an example
#     image_array = np.array(image)
#     blurred_image = PIL.Image.fromarray(image_array)  # Placeholder for inpainting logic
    
#     return blurred_image, transcribed_text

# Gradio interface
inputs = [
    #gr.Image(label="Upload an image"),
    gr.Audio(label="Record audio to describe what to remove")
]

outputs = [
    #gr.Image(label="Result Image"),
    gr.Textbox(label="Transcribed Text", type="text", lines=5)
]

title = "Image Inpainting Demo"
description = "Upload an image and record audio to describe what to remove. The result will be the inpainted image."

# Create the Gradio app
gr.Interface(fn=transcribe, inputs=inputs, outputs=outputs, title=title, description=description).launch()
