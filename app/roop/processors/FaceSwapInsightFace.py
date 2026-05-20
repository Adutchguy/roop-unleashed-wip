import roop.globals
import cv2
import numpy as np
import onnx
import onnxruntime

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path



class FaceSwapInsightFace():
    plugin_options:dict = None
    model_swap_insightface = None

    processorname = 'faceswap'
    type = 'swap'


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_swap_insightface is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            graph = onnx.load(model_path).graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            self.input_mean = 0.0
            self.input_std = 255.0
            #cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}            
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False
            self.model_swap_insightface = onnxruntime.InferenceSession(model_path, sess_options, providers=roop.globals.execution_providers)



    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        latent = source_face.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        # Use the standard run() API rather than io_binding.  io_binding with
        # bind_output() (no device_type) leaves output placement to TensorRT,
        # which registers a device type that copy_outputs_to_cpu() has no
        # transfer path for, raising:
        #   "There's no data transfer registered for copying tensors from
        #    Device:[DeviceType:0 ...] to Device:[DeviceType:0 ...]"
        # run() handles all device transfers internally and works correctly
        # across CPU, CUDA, and TensorRT execution providers.
        ort_outs = self.model_swap_insightface.run(
            None, {"target": temp_frame, "source": latent}
        )
        return ort_outs[0][0]


    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None


                



