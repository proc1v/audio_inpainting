import whisper

model = whisper.load_model("base")
result = model.transcribe("data/harvard.wav")
result2 = model.transcribe("data/jackhammer.wav")
print(result["text"])
print(result2["text"])

import os

segmentator = Segmentator()
lama_inpainter = Inpainter(InpaintBackend.LaMa)
sd_inpainter = Inpainter(InpaintBackend.SD, model_path="runwayml/stable-diffusion-inpainting")

ds_root = "/kaggle/input/audio-inpainting-ds"
path_to_image = os.path.join(ds_root, 'woman_dog_running.jpg')
original_image = PIL.Image.open(path_to_image)

masks = segmentator.generate_masks(path_to_image, ["dog"])
lama_result = lama_inpainter.inpaint(original_image, masks)
sd_result = sd_inpainter.inpaint(original_image, masks, ["white dragon"])