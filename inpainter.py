import numpy as np
import torch
import PIL
import cv2
from diffusers import StableDiffusionInpaintPipeline
from FastSAM.fastsam import FastSAM, FastSAMPrompt

class ImageInpainting:
    def __init__(self,
                 fastsam_path: str = "FastSAM.pt",
                 diffusion_path: str = "runwayml/stable-diffusion-inpainting",
                 config: dict = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentation_model = FastSAM(fastsam_path)#.to(self.device)
        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            diffusion_path,
            torch_dtype=torch.float16,
        ).to(self.device)
        
        # Default configuration
        default_config = {
            'imgsz': 1024,
            'conf': 0.4,
            'iou': 0.5,
            'dilation_radius': (27, 27),
            'guidance_scale': 7.5,
            'num_inference_steps': 50,
            'seed': 42,
            'mask_threshold': 100
        }
        
        # Update defaults with any configurations passed to constructor
        self.config = {**default_config, **(config if config is not None else {})}

    def generate_masks(self, image_path: str, prompts: list):
        # Get segmentation masks using configurations from self.config
        everything_results = self.segmentation_model(
            image_path,
            device=self.device,
            retina_masks=True,
            imgsz=self.config['imgsz'],
            conf=self.config['conf'],
            iou=self.config['iou'],
        )
        prompt_process = FastSAMPrompt(image_path, everything_results, device=self.device)
        masks = []
        for prompt in prompts:
            raw_mask = prompt_process.text_prompt(text=prompt)[0]
            kernel = np.ones(self.config['dilation_radius'], np.uint8)
            dilated_mask = cv2.dilate(raw_mask.astype(np.uint8), kernel, iterations=1)
            masks.append(dilated_mask)
        return masks

    def inpaint(self, image: PIL.Image.Image, masks: list, prompts: list = [""]):
        original_size = image.size
        result_images = []
        image = image.resize((512, 512))
        for mask in masks:
            if mask.sum() < self.config['mask_threshold']: continue
            mask_image = PIL.Image.fromarray(mask * 255).convert("RGB").resize((512, 512))
            inpainted_image = self._inpaint_single(image, mask_image, prompt="")
            print(f"mask size: {mask_image.size}")
            print(f"original size: {image.size}")
            print(f"inpainted_image size: {inpainted_image.size}")
            image = self._restore_unmasked(image, inpainted_image, mask_image)
        return image.resize(original_size)

    def _inpaint_single(self, image: PIL.Image.Image, mask_image: PIL.Image.Image, prompt: str = ""):
        # Use configurations from self.config for inpainting
        generator = torch.Generator(device=self.device).manual_seed(self.config['seed'])
        inpainted_images = self.inpaint_pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            guidance_scale=self.config['guidance_scale'],
            generator=generator,
            num_inference_steps=self.config['num_inference_steps'],
        ).images
        
        return inpainted_images[0]
        
    def _restore_unmasked(self, original: PIL.Image.Image, edited: PIL.Image.Image, mask: PIL.Image.Image):
        # restore the image outside of the mask
        edited_array = np.array(edited)
        mask_array =  np.array(mask)
        original_array = np.array(original)
        
        mask_array = mask_array.astype(bool)
        original_array[mask_array] = edited_array[mask_array]
        original_image = PIL.Image.fromarray(original_array)
        return original_image
