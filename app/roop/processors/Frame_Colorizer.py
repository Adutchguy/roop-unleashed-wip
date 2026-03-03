import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.utilities import resolve_relative_path
from roop.typing import Frame

class Frame_Colorizer():
    plugin_options:dict = None
    model_colorizer = None
    devicename = None
    prev_type = None

    processorname = 'deoldify'
    type = 'frame_colorizer'
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.prev_type is not None and self.prev_type != self.plugin_options["subtype"]:
            self.Release()
        self.prev_type = self.plugin_options["subtype"]
        if self.model_colorizer is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            if self.prev_type == "deoldify_artistic":
                model_path = resolve_relative_path('../models/Frame/deoldify_artistic.onnx')
            elif self.prev_type == "deoldify_stable":
                model_path = resolve_relative_path('../models/Frame/deoldify_stable.onnx')

            onnxruntime.set_default_logger_severity(3)
            self.model_colorizer = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_colorizer.get_inputs()
            model_outputs = self.model_colorizer.get_outputs()
            self.io_binding = self.model_colorizer.io_binding()
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)

    def Run(self, input_frame: Frame) -> Frame:
        temp_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_GRAY2RGB)
        temp_frame = cv2.resize(temp_frame, (256, 256))
        temp_frame = temp_frame.transpose((2, 0, 1))
        temp_frame = np.expand_dims(temp_frame, axis=0).astype(np.float32)
        self.io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        self.model_colorizer.run_with_iobinding(self.io_binding)
        ort_outs = self.io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs
        colorized_frame = result.transpose(1, 2, 0)
        colorized_frame = cv2.resize(colorized_frame, (input_frame.shape[1], input_frame.shape[0]))
        temp_blue_channel, _, _ = cv2.split(input_frame)
        colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
        colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_BGR2LAB)
        _, color_green_channel, color_red_channel = cv2.split(colorized_frame)
        colorized_frame = cv2.merge((temp_blue_channel, color_green_channel, color_red_channel))
        colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_LAB2BGR)
        return colorized_frame.astype(np.uint8)


    def Release(self):
        del self.model_colorizer
        self.model_colorizer = None
        del self.io_binding
        self.io_binding = None

