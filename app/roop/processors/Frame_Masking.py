import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.utilities import resolve_relative_path
from roop.typing import Frame

class Frame_Masking():
    plugin_options:dict = None
    model_masking = None
    devicename = None
    name = None

    processorname = 'removebg'
    type = 'frame_masking'
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_masking is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"]
            self.devicename = self.devicename.replace('mps', 'cpu')
            model_path = resolve_relative_path('../models/Frame/isnet-general-use.onnx')
            self.model_masking = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_masking.get_inputs()
            model_outputs = self.model_masking.get_outputs()
            self.io_binding = self.model_masking.io_binding()
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)

    def Run(self, temp_frame: Frame) -> Frame:
        # Pre process:Resize, BGR->RGB, float32 cast
        input_image = cv2.resize(temp_frame, (1024, 1024))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        mean = [0.5, 0.5, 0.5]
        std = [1.0, 1.0, 1.0]
        input_image = (input_image / 255.0 - mean) / std
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')
        
        self.io_binding.bind_cpu_input(self.model_inputs[0].name, input_image)
        self.model_masking.run_with_iobinding(self.io_binding)
        ort_outs = self.io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs
        # Post process:squeeze, Sigmoid, Normarize, uint8 cast
        mask = np.squeeze(result[0])
        min_value = np.min(mask)
        max_value = np.max(mask)
        mask = (mask - min_value) / (max_value - min_value)
        #mask = np.where(mask < score_th, 0, 1)
        #mask *= 255
        mask = cv2.resize(mask, (temp_frame.shape[1], temp_frame.shape[0]), interpolation=cv2.INTER_LINEAR)        
        mask = np.reshape(mask, [mask.shape[0],mask.shape[1],1])
        result = mask * temp_frame.astype(np.float32)
        return result.astype(np.uint8)



    def Release(self):
        del self.model_masking
        self.model_masking = None
        del self.io_binding
        self.io_binding = None

