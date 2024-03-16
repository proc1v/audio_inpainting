import shutil

import torch
import gradio as gr
import numpy as np
import PIL

from speech_recognizer import SpeechRecognizer
from llm_manager import LLM_Manager
from inpainter import Inpainter, InpaintBackend
from segmentator import Segmentator, SegmentBackend

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to process the image and audio inputs
def process_input(image, audio):
    print(device)
    # Speech recognition
    recognizer = SpeechRecognizer(device=device)
    transcribed_text = recognizer.transcribe(audio)
    
    print("#"*50)
    print("Transcribed Text:", transcribed_text)
    
    llm_manager = LLM_Manager()
    llm_output = llm_manager.run_chain(transcribed_text)
    
    print("#"*50)
    print("LLM Output:", llm_output)
    
    extracted_entities = llm_output['output']
    
    original_image = PIL.Image.fromarray(image)
    path_to_image = "temp.jpg"
    original_image.save(path_to_image)
    
    lang_segmentator = Segmentator(SegmentBackend.LangSAM)
    lang_masks = lang_segmentator.get_masks(path_to_image, extracted_entities)
    
    lama_inpainter = Inpainter(InpaintBackend.LaMa)
    lang_lama_result = lama_inpainter.inpaint(original_image, lang_masks)    
    
    shutil.rmtree(path_to_image)
    
    return lang_lama_result, str(extracted_entities)

# Gradio interface
inputs = [
    gr.Image(label="Upload an image"),
    gr.Audio(label="Record audio to describe what to remove")
]

outputs = [
    gr.Image(label="Result Image"),
    gr.Textbox(label="Transcribed Text", type="text", lines=5)
]

title = "Image Inpainting Demo"
description = "Upload an image and record audio to describe what to remove. The result will be the inpainted image."

with gr.Blocks() as demo:
    # Create the Gradio app
    gr.Interface(fn=process_input, inputs=inputs, outputs=outputs, title=title, description=description)
    
    
if __name__ == "__main__":
    demo.launch(share=True)
