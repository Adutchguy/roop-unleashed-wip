from typing import Any, List, Callable
import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path


# THREAD_LOCK = threading.Lock()


class Enhance_CodeFormer():
    model_codeformer = None

    plugin_options:dict = None

    processorname = 'codeformer'
    type = 'enhance'
    

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_codeformer is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            model_path = resolve_relative_path('../models/CodeFormer/CodeFormerv0.1.onnx')
            self.model_codeformer = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_codeformer.get_inputs()
            model_outputs = self.model_codeformer.get_outputs()
            self.io_binding = self.model_codeformer.io_binding()           
            self.io_binding.bind_cpu_input(self.model_inputs[1].name, np.array([0.5]))
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)


    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        input_size = temp_frame.shape[1]
        # preprocess
        temp_frame = cv2.resize(temp_frame, (512, 512), cv2.INTER_CUBIC)
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)
        
        self.io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame.astype(np.float32))
        self.model_codeformer.run_with_iobinding(self.io_binding)
        ort_outs = self.io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs
        
        # post-process
        result = result.transpose((1, 2, 0))

        un_min = -1.0
        un_max = 1.0
        result = np.clip(result, un_min, un_max)
        result = (result - un_min) / (un_max - un_min)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = (result * 255.0).round()
        scale_factor = int(result.shape[1] / input_size)       
        return result.astype(np.uint8), scale_factor


    def Release(self):
        del self.model_codeformer
        self.model_codeformer = None
        del self.io_binding
        self.io_binding = None

