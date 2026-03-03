import cv2 
import numpy as np
import onnxruntime
import roop.globals
import threading

from roop.utilities import resolve_relative_path
from roop.typing import Frame

class Frame_Upscale():
    plugin_options:dict = None
    model_upscale = None
    devicename = None
    prev_type = None

    processorname = 'upscale'
    type = 'frame_enhancer'

    THREAD_LOCK_UPSCALE = threading.Lock()


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.prev_type is not None and self.prev_type != self.plugin_options["subtype"]:
            self.Release()
        self.prev_type = self.plugin_options["subtype"]
        if self.model_upscale is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            if self.prev_type == "esrganx4":
                model_path = resolve_relative_path('../models/Frame/real_esrgan_x4.onnx')
                self.scale = 4
            elif self.prev_type == "esrganx2":
                model_path = resolve_relative_path('../models/Frame/real_esrgan_x2.onnx')
                self.scale = 2
            elif self.prev_type == "lsdirx4":
                model_path = resolve_relative_path('../models/Frame/lsdir_x4.onnx')
                self.scale = 4

            self.model_upscale = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_upscale.get_inputs()
            model_outputs = self.model_upscale.get_outputs()
            self.io_binding = self.model_upscale.io_binding()
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)

    def getProcessedResolution(self, width, height):
        return (width * self.scale, height * self.scale)

# borrowed from facefusion -> https://github.com/facefusion/facefusion
    def prepare_tile_frame(self, tile_frame : Frame) -> Frame:
        tile_frame = np.expand_dims(tile_frame[:, :, ::-1], axis = 0)
        tile_frame = tile_frame.transpose(0, 3, 1, 2)
        tile_frame = tile_frame.astype(np.float32) / 255
        return tile_frame


    def normalize_tile_frame(self, tile_frame : Frame) -> Frame:
        tile_frame = tile_frame.transpose(0, 2, 3, 1).squeeze(0) * 255
        tile_frame = tile_frame.clip(0, 255).astype(np.uint8)[:, :, ::-1]
        return tile_frame

    def create_tile_frames(self, input_frame : Frame, size):
        input_frame = np.pad(input_frame, ((size[1], size[1]), (size[1], size[1]), (0, 0)))
        tile_width = size[0] - 2 * size[2]
        pad_size_bottom = size[2] + tile_width - input_frame.shape[0] % tile_width
        pad_size_right = size[2] + tile_width - input_frame.shape[1] % tile_width
        pad_vision_frame = np.pad(input_frame, ((size[2], pad_size_bottom), (size[2], pad_size_right), (0, 0)))
        pad_height, pad_width = pad_vision_frame.shape[:2]
        row_range = range(size[2], pad_height - size[2], tile_width)
        col_range = range(size[2], pad_width - size[2], tile_width)
        tile_frames = []

        for row_frame in row_range:
            top = row_frame - size[2]
            bottom = row_frame + size[2] + tile_width
            for column_vision_frame in col_range:
                left = column_vision_frame - size[2]
                right = column_vision_frame + size[2] + tile_width
                tile_frames.append(pad_vision_frame[top:bottom, left:right, :])
        return tile_frames, pad_width, pad_height


    def merge_tile_frames(self, tile_frames, temp_width : int, temp_height : int, pad_width : int, pad_height : int, size) -> Frame:
        merge_frame = np.zeros((pad_height, pad_width, 3)).astype(np.uint8)
        tile_width = tile_frames[0].shape[1] - 2 * size[2]
        tiles_per_row = min(pad_width // tile_width, len(tile_frames))

        for index, tile_frame in enumerate(tile_frames):
            tile_frame = tile_frame[size[2]:-size[2], size[2]:-size[2]]
            row_index = index // tiles_per_row
            col_index = index % tiles_per_row
            top = row_index * tile_frame.shape[0]
            bottom = top + tile_frame.shape[0]
            left = col_index * tile_frame.shape[1]
            right = left + tile_frame.shape[1]
            merge_frame[top:bottom, left:right, :] = tile_frame
        merge_frame = merge_frame[size[1] : size[1] + temp_height, size[1]: size[1] + temp_width, :]
        return merge_frame


    def Run(self, temp_frame: Frame) -> Frame:
        size = (128, 8, 2)
        temp_height, temp_width = temp_frame.shape[:2]
        upscale_tile_frames, pad_width, pad_height = self.create_tile_frames(temp_frame, size)

        for index, tile_frame in enumerate(upscale_tile_frames):
            tile_frame = self.prepare_tile_frame(tile_frame)
            with self.THREAD_LOCK_UPSCALE:
                self.io_binding.bind_cpu_input(self.model_inputs[0].name, tile_frame)
                self.model_upscale.run_with_iobinding(self.io_binding)
                ort_outs = self.io_binding.copy_outputs_to_cpu()
                result = ort_outs[0]
            upscale_tile_frames[index] = self.normalize_tile_frame(result)
        final_frame = self.merge_tile_frames(upscale_tile_frames, temp_width * self.scale
                                                    , temp_height * self.scale
                                                    , pad_width * self.scale, pad_height * self.scale
                                                    , (size[0] * self.scale, size[1] * self.scale, size[2] * self.scale))
        return final_frame.astype(np.uint8)



    def Release(self):
        del self.model_upscale
        self.model_upscale = None
        del self.io_binding
        self.io_binding = None

