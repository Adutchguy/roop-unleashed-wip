from settings import Settings
from typing import List

source_path = None
target_path = None
output_path = None
target_folder_path = None
startup_args = None

cuda_device_id = 0
frame_processors: List[str] = []
keep_fps = None
keep_frames = None
autorotate_faces = None
vr_mode = None
skip_audio = None
wait_after_extraction = None
many_faces = None
use_batch = None
source_face_index = 0
target_face_index = 0
face_position = None
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads = None
headless = None
log_level = 'debug'
selected_enhancer = None
subsample_size = 256
face_swap_mode = 'DFL XSeg'
blend_ratio = 0.80
distance_threshold = 1
default_det_size = True

no_face_action = 1

processing = False

g_current_face_analysis = None
g_desired_face_analysis = None

FACE_ENHANCER = 'GPEN'

INPUT_FACESETS = []
TARGET_FACES = []


IMAGE_CHAIN_PROCESSOR = None
VIDEO_CHAIN_PROCESSOR = None
BATCH_IMAGE_CHAIN_PROCESSOR = None

CFG: Settings = None


