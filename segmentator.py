import numpy as np
import torch
from PIL import Image
import cv2
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from lang_sam import LangSAM
from enum import Enum, auto

class SegmentBackend(Enum):
    FastSAM = auto()
    LangSAM = auto()

class Segmentator:
    def __init__(self, backend: SegmentBackend = SegmentBackend.LangSAM, model_path: str = "FastSAM.pt", config: dict = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend
        if backend == SegmentBackend.FastSAM:
            self.model = FastSAM(model_path)#.to(self.device)
        else:
            self.model = LangSAM()

        default_config = {
            'imgsz': 1024,
            'conf': 0.4,
            'iou': 0.5,
            'dilation_kernel': (27, 27),
        }

        # Update defaults with any configurations passed to constructor
        self.config = {**default_config, **(config if config is not None else {})}
    
    def get_masks(self, image_path: str, prompts: list):
        if self.backend == SegmentBackend.LangSAM:
            return self._get_masks_langsam(image_path, prompts)
        else:
            return self._get_masks_fastsam(image_path, prompts)
        
    def _get_masks_langsam(self, image_path: str, prompts: list):
        masks = []
        for prompt in prompts:
            mask, _, _, _ = self.model.predict(original_image, prompt)
            np_mask = mask[0].numpy()
            kernel = np.ones(self.config['dilation_kernel'], np.uint8)
            dilated_mask = cv2.dilate(np_mask.astype(np.uint8), kernel, iterations=1)
            masks.append(dilated_mask)
        return masks
    
    def _get_masks_fastsam(self, image_path: str, prompts: list):
        everything_results = self.model(
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
            kernel = np.ones(self.config['dilation_kernel'], np.uint8)
            dilated_mask = cv2.dilate(raw_mask.astype(np.uint8), kernel, iterations=1)
            masks.append(dilated_mask)
        return masks