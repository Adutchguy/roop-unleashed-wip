import cv2
import numpy as np
import torch
import threading
from torchvision import transforms
from clip.clipseg import CLIPDensePredT
import numpy as np

from roop.typing import Frame

THREAD_LOCK_CLIP = threading.Lock()


class Mask_Clip2Seg():
    plugin_options:dict = None
    model_clip = None

    processorname = 'clip2seg'
    type = 'mask'


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_clip is None:
            self.model_clip = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            self.model_clip.eval();
            self.model_clip.load_state_dict(torch.load('models/CLIP/rd64-uni-refined.pth', map_location=torch.device('cpu')), strict=False)

        device = torch.device(self.plugin_options["devicename"])
        self.model_clip.to(device)


    def Run(self, img1, keywords:str) -> Frame:
        if keywords is None or len(keywords) < 1 or img1 is None:
            return img1

        orig_h, orig_w = img1.shape[:2]
        mask_blur = 5
        clip_blur = 5
        thresh    = 0.5

        # Border validity mask at 256×256 (matches model output size).
        img_mask_256 = np.full((256, 256), 0, dtype=np.float32)
        mask_border  = 1
        img_mask_256 = cv2.rectangle(img_mask_256,
                                     (mask_border, mask_border),
                                     (256 - mask_border, 256 - mask_border),
                                     (255, 255, 255), -1)
        img_mask_256 = cv2.GaussianBlur(img_mask_256, (mask_blur*2+1, mask_blur*2+1), 0)
        img_mask_256 /= 255

        # #4: Feed the original-resolution image to the transform so the
        # quality-critical resize to 256 is handled by PyTorch (bilinear on
        # higher-detail input) rather than a preliminary cv2 downsample.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((256, 256)),
        ])
        img = transform(img1).unsqueeze(0)

        prompts = keywords.split(',')
        with THREAD_LOCK_CLIP:
            with torch.no_grad():
                preds = self.model_clip(img.repeat(len(prompts), 1, 1, 1), prompts)[0]
        clip_mask = torch.sigmoid(preds[0][0])
        for i in range(len(prompts) - 1):
            clip_mask += torch.sigmoid(preds[i+1][0])

        clip_mask = clip_mask.data.cpu().numpy()   # 256×256 float

        # #4: Upscale CLIP output to the original crop resolution BEFORE
        # thresholding and morphological ops.  At 512×512 a 5-px dilation covers
        # ~10px equivalent detail vs. a blurry ~2.5px at 256 — significantly
        # better at capturing fine hair strands and thin wet-artifact edges.
        if orig_h != 256 or orig_w != 256:
            clip_mask = cv2.resize(clip_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            img_mask  = cv2.resize(img_mask_256, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            scale = max(orig_w, orig_h) / 256.0
            dil_k  = max(3, int(5 * scale))
            blur_k = max(clip_blur*2+1, int(clip_blur * 2 * scale))
            if blur_k % 2 == 0:
                blur_k += 1   # kernel must be odd
        else:
            img_mask = img_mask_256
            dil_k    = 5
            blur_k   = clip_blur * 2 + 1

        np.clip(clip_mask, 0, 1, out=clip_mask)
        clip_mask[clip_mask >  thresh] = 1.0
        clip_mask[clip_mask <= thresh] = 0.0

        kernel    = np.ones((dil_k, dil_k), np.float32)
        clip_mask = cv2.dilate(clip_mask, kernel, iterations=1)
        clip_mask = cv2.GaussianBlur(clip_mask, (blur_k, blur_k), 0)

        img_mask *= clip_mask
        img_mask[img_mask < 0.0] = 0.0
        return img_mask
       


    def Release(self):
        self.model_clip = None

