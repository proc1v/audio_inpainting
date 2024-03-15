import numpy as np
import torch
from PIL import Image
import cv2
from FastSAM.fastsam import FastSAM, FastSAMPrompt

class Segmentator:
    def __init__(self, model_path: str = "FastSAM.pt", config: dict = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentation_model = FastSAM(model_path)#.to(self.device)

        default_config = {
            'imgsz': 1024,
            'conf': 0.4,
            'iou': 0.5,
            'dilation_kernel': (27, 27),
        }

        # Update defaults with any configurations passed to constructor
        self.config = {**default_config, **(config if config is not None else {})}

    def generate_masks(self, image_path: str, prompts: list):
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
            print(prompt)
            raw_mask = prompt_process.text_prompt(text=prompt)[0]
            kernel = np.ones(self.config['dilation_kernel'], np.uint8)
            dilated_mask = cv2.dilate(raw_mask.astype(np.uint8), kernel, iterations=1)
            masks.append(dilated_mask)
        return masks