"""
FaceSwapGHOST.py — GHOST face-swap processor for roop-unleashed
================================================================
Wraps the GHOST (Generative High-fidelity face swap with Occlusion-aware
Transformer) ONNX models from FaceFusion's model repository.

Required model files (place both in <roop-unleashed>/app/models/):
-------------------------------------------------------------------
1. One of:
       ghost_1_256.onnx  (~340 MB, fastest)
       ghost_2_256.onnx  (~515 MB, balanced)  ← default
       ghost_3_256.onnx  (~739 MB, best quality)

   Download from:
       https://huggingface.co/facefusion/models-3.0.0

2. crossface_ghost.onnx  (embedding converter, ~small)

   Download from:
       https://huggingface.co/facefusion/models-3.4.0

Pipeline
--------
  source_face.embedding (512-d)
       ↓  crossface_ghost.onnx  (input='input', output converted embedding)
  source_embedding_converted (512-d)
       ↓  ghost_X_256.onnx  (inputs: 'target'=crop, 'source'=embedding)
  swapped crop (1, 3, 256, 256)

Normalization
-------------
GHOST uses mean=0.5 std=0.5 (i.e. range [-1, 1]).
ProcessMgr.prepare_crop_frame / normalize_swap_frame handle this when
_active_swap_model_type() returns 'ghost'.
"""

import os
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path


# Ghost model candidates, tried in order
_GHOST_MODEL_NAMES = [
    'ghost_2_256.onnx',
    'ghost_1_256.onnx',
    'ghost_3_256.onnx',
    'ghost.onnx',          # legacy / manual rename fallback
]
_CROSSFACE_MODEL_NAME = 'crossface_ghost.onnx'


class FaceSwapGHOST:
    plugin_options: dict = None
    model = None            # ghost_X_256.onnx session
    model_converter = None  # crossface_ghost.onnx session
    model_output_size: int = 256

    processorname = 'ghost'
    type = 'swap'

    # ------------------------------------------------------------------ init

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options.get('devicename') != plugin_options.get('devicename'):
                self.Release()

        self.plugin_options = plugin_options

        if self.model is None:
            self._load_swap_model()

        if self.model_converter is None:
            self._load_converter_model()

    def _load_swap_model(self):
        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_cpu_mem_arena = False

        model_path = None
        for name in _GHOST_MODEL_NAMES:
            candidate = resolve_relative_path(f'../models/{name}')
            if os.path.isfile(candidate):
                model_path = candidate
                print(f"[GHOST] Using swap model: {name}")
                break

        if model_path is None:
            candidates_str = ', '.join(_GHOST_MODEL_NAMES)
            raise FileNotFoundError(
                f"[GHOST] No swap model found. Place one of [{candidates_str}] "
                f"in app/models/. Download from: "
                f"https://huggingface.co/facefusion/models-3.0.0"
            )

        self.model = onnxruntime.InferenceSession(
            model_path, sess_options,
            providers=roop.globals.execution_providers,
        )

        # Detect output size from the model graph
        for inp in self.model.get_inputs():
            if inp.name == 'target':
                h = inp.shape[2]
                if isinstance(h, int) and h > 0:
                    self.model_output_size = h
                break

        print(f"[GHOST] Swap model loaded — output size: {self.model_output_size}px")

    def _load_converter_model(self):
        converter_path = resolve_relative_path(f'../models/{_CROSSFACE_MODEL_NAME}')
        if not os.path.isfile(converter_path):
            raise FileNotFoundError(
                f"[GHOST] Embedding converter not found at app/models/{_CROSSFACE_MODEL_NAME}. "
                f"Download from: https://huggingface.co/facefusion/models-3.4.0"
            )

        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        self.model_converter = onnxruntime.InferenceSession(
            converter_path, sess_options,
            providers=roop.globals.execution_providers,
        )
        print(f"[GHOST] Embedding converter loaded: {_CROSSFACE_MODEL_NAME}")

    # ------------------------------------------------------------------ run

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        """
        Parameters
        ----------
        temp_frame : float32 array, shape (1, 3, H, W), range [-1, 1].
                     Prepared by ProcessMgr.prepare_crop_frame (model_type='ghost').
        source_face : insightface Face object with ``embedding`` attribute.

        Returns
        -------
        float32 array, shape (3, H, W), range [-1, 1].
        """
        # Step 1 — convert raw ArcFace embedding via crossface_ghost.onnx
        raw_embedding = source_face.embedding.reshape(-1, 512).astype(np.float32)
        converted = self.model_converter.run(None, {'input': raw_embedding})[0]
        source_latent = converted.reshape(1, -1).astype(np.float32)

        # Step 2 — run ghost swap model
        result = self.model.run(
            None,
            {
                'target': temp_frame,
                'source': source_latent,
            }
        )[0]

        # result shape: (1, 3, H, W) — drop batch dim
        return result[0]

    # ------------------------------------------------------------------ release

    def Release(self):
        del self.model
        del self.model_converter
        self.model = None
        self.model_converter = None
