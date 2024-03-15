from simple_lama_inpainting import SimpleLama
from PIL import Image
from enum import Enum, auto
import numpy as np
import torch
import cv2
from diffusers import StableDiffusionInpaintPipeline

class InpaintBackend(Enum):
    LaMa = auto()
    SD = auto()
        
class Inpainter:    
    def __init__(self, 
                 backend: InpaintBackend = InpaintBackend.LaMa, 
                 model_path:str = None, 
                 config: dict = None):
        default_config = {
            'guidance_scale': 7.5,
            'num_inference_steps': 50,
            'seed': 42,
            'mask_threshold': 100
        }
        self.config = {**default_config, **(config if config is not None else {})}
        self.backend = backend
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.backend == InpaintBackend.LaMa:
            self.pipeline = SimpleLama()
        else:
            if not model_path:
                raise ValueError("For backend==InpainterBackend.SD, you should specify the model_path.")
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(self.device)
    
    
    def inpaint(self, image: Image.Image, masks: list, prompts: list = []):
        if self.backend == InpaintBackend.LaMa:
            return self._lama_inpaint(image, masks)
        else:
            return self._sd_inpaint(image, masks, prompts)
    
    def _lama_inpaint(self, image: Image.Image, masks: list):
        result_image = image
        for mask in masks:
            single_channel_mask = Image.fromarray(mask).convert('L')
            result_image = self.pipeline(result_image, single_channel_mask)
        return result_image

    def _sd_inpaint(self, image: Image.Image, masks: list, prompts: list = []):
        original_size = image.size
        result_images = []
        image = image.resize((512, 512))
        prompts = prompts if len(prompts) == len(masks) else [""] * len(masks)
        for mask, prompt in zip(masks, prompts):
            if mask.sum() < self.config['mask_threshold']: continue
            mask_image = Image.fromarray(mask * 255).convert("RGB").resize((512, 512))
            cut_image = self._cut_mask(image, mask_image)
            inpainted_image = self._inpaint_single(cut_image, mask_image, prompt=prompt)
            image = self._restore_unmasked(image, inpainted_image, mask_image)
        return image.resize(original_size)

    def _inpaint_single(self, image: Image.Image, mask_image: Image.Image, prompt: str = ""):
        # Use configurations from self.config for inpainting
        generator = torch.Generator(device=self.device).manual_seed(self.config['seed'])
        inpainted_images = self.pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            guidance_scale=self.config['guidance_scale'],
            generator=generator,
            num_inference_steps=self.config['num_inference_steps'],
        ).images
        
        return inpainted_images[0]
        
    def _restore_unmasked(self, original: Image.Image, edited: Image.Image, mask: Image.Image):
        # restore the image outside of the mask
        edited_array = np.array(edited)
        mask_array =  np.array(mask)
        original_array = np.array(original)
        
        mask_array = mask_array.astype(bool)
        original_array[mask_array] = edited_array[mask_array]
        original_image = Image.fromarray(original_array)
        return original_image
    
    def _cut_mask(self, original: Image.Image, mask: Image.Image):
        mask_array =  np.array(mask)
        original_array = np.array(original)
        
        mask_array = mask_array.astype(bool)
        original_array[mask_array] = 0
        original_image = Image.fromarray(original_array)
        return original_image
