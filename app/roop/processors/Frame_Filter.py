import cv2
import numpy as np

from roop.typing import Frame

class Frame_Filter():
    processorname = 'generic_filter'
    type = 'frame_processor'

    plugin_options:dict = None

    c64_palette = np.array([
            [0, 0, 0],
            [255, 255, 255],
            [0x81, 0x33, 0x38],
            [0x75, 0xce, 0xc8],
            [0x8e, 0x3c, 0x97],
            [0x56, 0xac, 0x4d],
            [0x2e, 0x2c, 0x9b],
            [0xed, 0xf1, 0x71],
            [0x8e, 0x50, 0x29],
            [0x55, 0x38, 0x00],
            [0xc4, 0x6c, 0x71],
            [0x4a, 0x4a, 0x4a],
            [0x7b, 0x7b, 0x7b],
            [0xa9, 0xff, 0x9f],
            [0x70, 0x6d, 0xeb],
            [0xb2, 0xb2, 0xb2]
        ])


    def RenderC64Screen(self, image):
        # Simply round the color values to the nearest color in the palette
        image = cv2.resize(image,(320,200))
        palette = self.c64_palette / 255.0  # Normalize palette
        img_normalized = image  / 255.0  # Normalize image

        # Calculate the index in the palette that is closest to each pixel in the image
        indices = np.sqrt(((img_normalized[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=3)).argmin(axis=2)
        # Map the image to the palette colors
        mapped_image = palette[indices]
        return (mapped_image * 255).astype(np.uint8)  # Denormalize and return the image


    def RenderDetailEnhance(self, image):
        return cv2.detailEnhance(image)

    def RenderStylize(self, image):
        return cv2.stylization(image)
    
    def RenderPencilSketch(self, image):
        imgray, imout = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return imout
    
    def RenderCartoon(self, image):
        numDownSamples = 2 # number of downscaling steps
        numBilateralFilters = 7  # number of bilateral filtering steps

        img_color = image
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        if img_color.shape != image.shape:
            img_color = cv2.resize(img_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)        
        if img_color.shape != img_edge.shape:
            img_edge = cv2.resize(img_edge, (img_color.shape[1], img_color.shape[0]), interpolation=cv2.INTER_LINEAR)        
        return cv2.bitwise_and(img_color, img_edge)
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()
        self.plugin_options = plugin_options

    def Run(self, temp_frame: Frame) -> Frame:
        subtype = self.plugin_options["subtype"]
        if subtype == "stylize":
            return self.RenderStylize(temp_frame).astype(np.uint8)
        if subtype == "detailenhance":
            return self.RenderDetailEnhance(temp_frame).astype(np.uint8)
        if subtype == "pencil":
            return self.RenderPencilSketch(temp_frame).astype(np.uint8)
        if subtype == "cartoon":
            return self.RenderCartoon(temp_frame).astype(np.uint8)
        if subtype == "C64":
            return self.RenderC64Screen(temp_frame).astype(np.uint8)


    def Release(self):
        pass

    def getProcessedResolution(self, width, height):
        if self.plugin_options["subtype"] == "C64":
            return (320,200)
        return None

