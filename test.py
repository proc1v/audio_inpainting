import os

from inpainter import Inpainter, InpaintBackend
from segmentator import Segmentator, SegmentBackend
from PIL import Image

# Utils
from PIL import Image
import matplotlib.pyplot as plt

def image_grid(imgs: list[Image.Image], rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def plot_np(np_array):
    
    plt.imshow(np_array)
    plt.axis('off')  # Optional: Removes the axis for a cleaner image
    plt.show()

fast_segmentator = Segmentator(SegmentBackend.FastSAM)
lang_segmentator = Segmentator(SegmentBackend.LangSAM)
lama_inpainter = Inpainter(InpaintBackend.LaMa)
sd_inpainter = Inpainter(InpaintBackend.SD, model_path="runwayml/stable-diffusion-inpainting")
ds_root = "/kaggle/input/audio-inpainting-ds"
path_to_image = os.path.join(ds_root, 'woman_dog_running.jpg')
original_image = Image.open(path_to_image)

fast_masks = fast_segmentator.get_masks(path_to_image, ["dog", "woman"])
lang_masks = lang_segmentator.get_masks(path_to_image, ["dog", "woman"])

fast_lama_result = lama_inpainter.inpaint(original_image, fast_masks)
fast_sd_result = sd_inpainter.inpaint(original_image, fast_masks, [""])

lang_lama_result = lama_inpainter.inpaint(original_image, lang_masks)
lang_sd_result = sd_inpainter.inpaint(original_image, lang_masks)

image_grid([original_image, 
            fast_lama_result, fast_sd_result, 
            lang_lama_result, lang_sd_result], 5, 1)