#!/usr/bin/env python3

import os
import sys
import shutil
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from typing import List
import platform
import signal
import torch

try:
    import tensorrt  # registers TensorRT DLL paths on Windows so onnxruntime can find them
except ImportError:
    pass

import onnxruntime as ort
available_providers = ort.get_available_providers()
print("Available ONNX providers at startup:", available_providers)  # Debug

import pathlib
import argparse

from time import time
from roop.utilities import print_cuda_info
import roop.globals
import roop.metadata
import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.capturer import get_video_frame_total, release_video


clip_text = None

call_display_ui = None

process_mgr = None
_preview_process_mgr = None   # dedicated instance for live_swap — never shared with batch


if 'ROCMExecutionProvider' in roop.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    roop.globals.headless = False

    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('--server_share', help='Public server', dest='server_share', action='store_true', default=False)
    program.add_argument('--cuda_device_id', help='Index of the cuda gpu to use', dest='cuda_device_id', type=int, default=0)
    roop.globals.startup_args = program.parse_args()
    # Always enable all processors when using GUI
    roop.globals.frame_processors = ['face_swapper', 'face_enhancer']


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    import onnxruntime
    list_providers = [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]
    
    try:
        for i in range(len(list_providers)):
            if list_providers[i] == 'CUDAExecutionProvider':
                list_providers[i] = ('CUDAExecutionProvider', {'device_id': roop.globals.cuda_device_id})
                torch.cuda.set_device(roop.globals.cuda_device_id)
            elif list_providers[i] == 'TensorrtExecutionProvider':
                trt_cache = str(pathlib.Path(__file__).parent.parent / 'models' / 'trt_cache')
                os.makedirs(trt_cache, exist_ok=True)
                list_providers[i] = ('TensorrtExecutionProvider', {
                    'device_id': roop.globals.cuda_device_id,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': trt_cache,
                })
    except:
        pass

    return list_providers
    
# Force GPU if available
# roop.globals.execution_providers = decode_execution_providers(['cuda'])
# print("Forced execution providers:", roop.globals.execution_providers)  # Debug

def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    import onnxruntime
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in roop.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in roop.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))



def release_resources() -> None:
    import gc
    from roop.face_util import release_face_analyser
    global process_mgr, _preview_process_mgr

    release_face_analyser()
    if process_mgr is not None:
        process_mgr.release_resources()
        process_mgr = None
    if _preview_process_mgr is not None:
        _preview_process_mgr.release_resources()
        _preview_process_mgr = None

    gc.collect()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                with torch.cuda.device(roop.globals.cuda_device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        except Exception:
            pass


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    
    download_directory_path = util.resolve_relative_path('../models')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx'])
    util.conditional_download(download_directory_path, ['https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx'])
    download_directory_path = util.resolve_relative_path('../models/CLIP')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth'])
    download_directory_path = util.resolve_relative_path('../models/CodeFormer')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx'])
    download_directory_path = util.resolve_relative_path('../models/Frame')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_artistic.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_stable.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/isnet-general-use.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x4.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x2.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/lsdir_x4.onnx'])

    print_cuda_info()  # Debug CUDA during pre-check


    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True

def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)




def start() -> None:
    if roop.globals.headless:
        print('Headless mode currently unsupported - starting UI!')
        # faces = extract_face_images(roop.globals.source_path,  (False, 0))
        # roop.globals.INPUT_FACES.append(faces[roop.globals.source_face_index])
        # faces = extract_face_images(roop.globals.target_path,  (False, util.has_image_extension(roop.globals.target_path)))
        # roop.globals.TARGET_FACES.append(faces[roop.globals.target_face_index])
        # if 'face_enhancer' in roop.globals.frame_processors:
        #     roop.globals.selected_enhancer = 'GFPGAN'
       
    # FIX: was batch_process_regular(None, False, None) — only 3 args for a 10-param function.
    # Headless mode is unsupported in this fork; log and fall through to UI launch.
    print('Headless batch processing is not implemented - falling through to UI.')


def get_processing_plugins(masking_engine, swap_model='inswapper'):
    """Build the processor dict for ProcessOptions.

    swap_model: 'inswapper' uses FaceSwapInsightFace (default);
                'ghost'     uses FaceSwapGHOST (requires app/models/ghost.onnx).
    """
    if swap_model == 'ghost':
        processors = {"ghost": {}}
    else:
        processors = {"faceswap": {}}
    if masking_engine is not None:
        processors.update({masking_engine: {}})
    
    if roop.globals.selected_enhancer == 'GFPGAN':
        processors.update({"gfpgan": {}})
    elif roop.globals.selected_enhancer == 'Codeformer':
        processors.update({"codeformer": {}})
    elif roop.globals.selected_enhancer == 'DMDNet':
        processors.update({"dmdnet": {}})
    elif roop.globals.selected_enhancer == 'GPEN':
        processors.update({"gpen": {}})
    elif roop.globals.selected_enhancer == 'Restoreformer++':
        processors.update({"restoreformer++": {}})
    return processors


def get_face_crop_from_frame(frame_bgr) -> str:
    """Return a base64 PNG data-URL of the canonical 512×512 aligned face crop from *frame_bgr*.

    Replicates the same autorotation pre-processing that ProcessMgr.process_face uses, so
    the crop shown in the Frame Editor mask modal exactly matches the coordinate space the
    processor operates in.  Returns empty string when no face is detected.
    """
    import base64 as _b64
    import cv2 as _cv2
    from roop.face_util import get_first_face, align_crop, rotate_anticlockwise, rotate_clockwise

    if frame_bgr is None:
        return ""

    def _rotation_action(face, frame):
        bbox_w = face.bbox[2] - face.bbox[0]
        bbox_h = face.bbox[3] - face.bbox[1]
        if bbox_w <= bbox_h:
            return None
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            forehead_x = face.landmark_2d_106[72][0]
            chin_x     = face.landmark_2d_106[0][0]
            if chin_x < forehead_x:
                return "rotate_anticlockwise"
            if forehead_x < chin_x:
                return "rotate_clockwise"
        fh, fw = frame.shape[:2]
        bbox_cx = face.bbox[0] + bbox_w / 2.0
        return "rotate_anticlockwise" if bbox_cx >= fw / 2.0 else "rotate_clockwise"

    face = get_first_face(frame_bgr)
    if face is None or not hasattr(face, 'kps') or face.kps is None:
        return ""

    frame = frame_bgr.copy()
    if roop.globals.autorotate_faces:
        action = _rotation_action(face, frame)
        if action is not None:
            x0, y0, x1, y1 = face.bbox.astype(int)
            offs = int(max(x1 - x0, y1 - y0) * 0.25)
            x0m = max(0, x0 - offs); y0m = max(0, y0 - offs)
            x1m = min(frame.shape[1], x1 + offs); y1m = min(frame.shape[0], y1 + offs)
            cut = frame[y0m:y1m, x0m:x1m]
            if action == "rotate_anticlockwise":
                cut = rotate_anticlockwise(cut)
            else:
                cut = rotate_clockwise(cut)
            rotface = get_first_face(cut)
            if rotface is not None and hasattr(rotface, 'kps') and rotface.kps is not None:
                face  = rotface
                frame = cut

    crop, _ = align_crop(frame, face.kps, 512)
    ok, buf = _cv2.imencode('.png', crop)
    if not ok:
        return ""
    return "data:image/png;base64," + _b64.b64encode(buf.tobytes()).decode('utf-8')


def live_swap(frame, options):
    global _preview_process_mgr

    if frame is None:
        return frame

    if _preview_process_mgr is None:
        _preview_process_mgr = ProcessMgr(None)

    _preview_process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)
    newframe = _preview_process_mgr.process_frame(frame)
    if newframe is None:
        return frame
    return newframe


def _parse_per_frame_masks(json_str: str) -> dict:
    """Parse the JSON string from mask_per_frame_store.

    Supports two formats:
    - New: {"frame": {"facesetIdx": maskData, ...}, ...}
    - Old: {"frame": maskData, ...}  — backwards compat, wrapped as {"0": maskData}

    Returns {int_frame_num: {int_faceset_idx: maskData}}.
    """
    import json as _json
    if not json_str:
        return {}
    try:
        raw = _json.loads(json_str)
        if not isinstance(raw, dict):
            return {}
        result = {}
        for k, v in raw.items():
            if not k.isdigit() or not isinstance(v, dict):
                continue
            frame_num = int(k)
            # Detect old flat format: has 'exclude', 'include', or 'canonical' at top level
            is_old_flat = any(x in v for x in ('exclude', 'include', 'canonical'))
            if is_old_flat:
                result[frame_num] = {0: v}
            else:
                per_faceset = {int(fk): fv for fk, fv in v.items()
                               if fk.isdigit() and isinstance(fv, dict)}
                if per_faceset:
                    result[frame_num] = per_faceset
        return result
    except Exception:
        return {}


def _reprocess_custom_mask_frames(temp_frame_paths: list, orig_frame_paths: list,
                                   per_frame_masks: dict, masking_engine, new_clip_text: str,
                                   num_swap_steps: int, restore_original_mouth: bool,
                                   selected_index: int, use_3d_recon: bool,
                                   use_source_bank: bool = False,
                                   use_frontalization: bool = False,
                                   frontalization_threshold: float = 25.0,
                                   swap_model: str = 'inswapper') -> None:
    """Re-process frames that have a custom per-frame mask.

    Strategy:
    - temp_frame_paths contains the already-swapped frames (global-mask run).
    - orig_frame_paths are the pre-swap originals saved by save_original_frames().
    - For each frame number in per_frame_masks, re-run live_swap on the original
      with the custom mask and overwrite the corresponding temp frame.

    Frame numbers in per_frame_masks are 1-based to match the UI slider / JS.
    The temp / orig path lists are 0-based.
    """
    if not per_frame_masks or not orig_frame_paths:
        return

    import cv2 as _cv2
    import json as _json

    plugins = get_processing_plugins(masking_engine, swap_model=swap_model)

    # per_frame_masks: {int_frame_num: {int_faceset_idx: maskData}}
    for frame_num_1, faceset_masks in per_frame_masks.items():
        idx = frame_num_1 - 1          # convert 1-based → 0-based list index
        if idx < 0 or idx >= len(orig_frame_paths):
            continue
        orig_path = orig_frame_paths[idx]
        out_path  = temp_frame_paths[idx] if idx < len(temp_frame_paths) else orig_path

        orig_bgr = _cv2.imread(orig_path)
        if orig_bgr is None:
            print(f"[per-frame mask] could not read original {orig_path}")
            continue

        # Build combined per-faceset mask JSON: {"0": maskData, "1": maskData, ...}
        # ProcessMgr.initialize detects digit-string top-level keys as new format.
        combined_mask = {str(fi): fd for fi, fd in faceset_masks.items()
                         if isinstance(fd, dict)}
        mask_json_str = _json.dumps(combined_mask) if combined_mask else None

        options = ProcessOptions(
            plugins,
            roop.globals.distance_threshold,
            roop.globals.blend_ratio,
            roop.globals.face_swap_mode,
            selected_index,
            new_clip_text,
            mask_json_str,
            num_swap_steps,
            roop.globals.subsample_size,
            False,
            restore_original_mouth,
            use_3d_recon=use_3d_recon,
            use_source_bank=use_source_bank,
            use_frontalization=use_frontalization,
            frontalization_threshold=frontalization_threshold,
            swap_model=swap_model,
        )
        result = live_swap(orig_bgr, options)
        if result is not None:
            _cv2.imwrite(out_path, result)
            print(f"[per-frame mask] frame {frame_num_1} reprocessed → {os.path.basename(out_path)}")


def batch_process_regular(output_method, files:list[ProcessEntry], masking_engine:str, new_clip_text:str, use_new_method, imagemask, restore_original_mouth, num_swap_steps, progress, selected_index = 0, use_3d_recon=False, mask_per_frame_json="",
                          use_source_bank=False, use_frontalization=False,
                          frontalization_threshold=25.0, swap_model='inswapper') -> None:
    global clip_text, process_mgr

    release_resources()
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    # imagemask is a JSON string produced by the canvas masking modal
    # (keys: "include" and/or "exclude", values: grayscale PNG data-URLs).
    # ProcessMgr.initialize decodes it into include_mask / exclude_mask arrays.
    if len(roop.globals.INPUT_FACESETS) <= selected_index:
        selected_index = 0
    options = ProcessOptions(get_processing_plugins(masking_engine, swap_model=swap_model),
                              roop.globals.distance_threshold, roop.globals.blend_ratio,
                              roop.globals.face_swap_mode, selected_index, new_clip_text, imagemask, num_swap_steps,
                              roop.globals.subsample_size, False, restore_original_mouth,
                              use_3d_recon=use_3d_recon,
                              use_source_bank=use_source_bank,
                              use_frontalization=use_frontalization,
                              frontalization_threshold=frontalization_threshold,
                              swap_model=swap_model)
    process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)

    # Stash per-frame mask map and batch options on globals so batch_process can access them
    roop.globals.mask_per_frame = _parse_per_frame_masks(mask_per_frame_json)
    roop.globals._batch_selected_index    = selected_index
    roop.globals._batch_clip_text         = new_clip_text
    roop.globals._batch_num_steps         = num_swap_steps
    roop.globals._batch_restore_mouth     = restore_original_mouth
    roop.globals._batch_use_3d_recon      = use_3d_recon
    roop.globals._batch_use_source_bank   = use_source_bank
    roop.globals._batch_use_frontalization= use_frontalization
    roop.globals._batch_front_threshold   = frontalization_threshold
    roop.globals._batch_swap_model        = swap_model

    batch_process(output_method, files, use_new_method)
    return

def batch_process_with_options(files:list[ProcessEntry], options, progress):
    global clip_text, process_mgr

    release_resources()
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)
    roop.globals.keep_frames = False
    roop.globals.wait_after_extraction = False
    roop.globals.skip_audio = False
    batch_process("Files", files, True)



def batch_process(output_method, files:list[ProcessEntry], use_new_method) -> None:
    global clip_text, process_mgr

    roop.globals.processing = True

    # limit threads for some providers
    max_threads = suggest_execution_threads()
    if max_threads == 1:
        roop.globals.execution_threads = 1

    imagefiles:list[ProcessEntry] = []
    videofiles:list[ProcessEntry] = []
           
    update_status('Sorting videos/images')


    for index, f in enumerate(files):
        fullname = f.filename
        if util.is_video(fullname) or util.has_extension(fullname, ['gif']) or util.is_animated_webp(fullname):
            destination = util.get_destfilename_from_path(fullname, roop.globals.output_path, f'__temp.{roop.globals.CFG.output_video_format}')
            f.finalname = destination
            videofiles.append(f)

        elif util.has_image_extension(fullname):
            destination = util.get_destfilename_from_path(fullname, roop.globals.output_path, f'.{roop.globals.CFG.output_image_format}')
            destination = util.replace_template(destination, index=index)
            pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
            f.finalname = destination
            imagefiles.append(f)



    if(len(imagefiles) > 0):
        update_status('Processing image(s)')
        origimages = []
        fakeimages = []
        for f in imagefiles:
            origimages.append(f.filename)
            fakeimages.append(f.finalname)

        process_mgr.run_batch(origimages, fakeimages, roop.globals.execution_threads)
        origimages.clear()
        fakeimages.clear()

    if(len(videofiles) > 0):
        for index,v in enumerate(videofiles):
            if not roop.globals.processing:
                end_processing('Processing stopped!')
                return
            fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
            if v.endframe == 0:
                v.endframe = get_video_frame_total(v.filename)

            is_streaming_only = output_method == "Virtual Camera"
            if is_streaming_only == False:
                update_status(f'Creating {os.path.basename(v.finalname)} with {fps} FPS...')

            start_processing = time()
            _has_per_frame_masks = bool(getattr(roop.globals, 'mask_per_frame', {}))
            if (is_streaming_only == False and roop.globals.keep_frames) or not use_new_method or (is_streaming_only == False and _has_per_frame_masks):
                util.create_temp(v.filename)
                update_status('Extracting frames...')
                extraction_ok = ffmpeg.extract_frames(v.filename,v.startframe,v.endframe, fps)
                if not roop.globals.processing:
                    end_processing('Processing stopped!')
                    return

                temp_frame_paths = util.get_temp_frame_paths(v.filename)
                if not temp_frame_paths:
                    # Frame extraction produced no output — ffmpeg likely failed above.
                    # Log and skip this video rather than crashing on temp_frame_paths[0].
                    update_status(f'Frame extraction failed for {os.path.basename(v.filename)}, skipping...')
                    continue

                # Save unswapped originals BEFORE run_batch overwrites them in-place.
                # Needed for both keep_frames mode (Frame Editor) and per-frame mask re-processing.
                per_frame_masks = getattr(roop.globals, 'mask_per_frame', {})
                needs_originals = roop.globals.keep_frames or bool(per_frame_masks)
                if needs_originals:
                    util.save_original_frames(v.filename)
                process_mgr.run_batch(temp_frame_paths, temp_frame_paths, roop.globals.execution_threads)
                if not roop.globals.processing:
                    end_processing('Processing stopped!')
                    return

                # Re-process any frames that have custom per-frame masks.
                if per_frame_masks:
                    update_status('Applying per-frame masks...')
                    orig_paths = util.get_temp_frame_paths_from_dir(util.get_frames_orig_path(v.filename))
                    _reprocess_custom_mask_frames(
                        temp_frame_paths, orig_paths, per_frame_masks,
                        masking_engine=None,
                        new_clip_text=getattr(roop.globals, '_batch_clip_text', ''),
                        num_swap_steps=getattr(roop.globals, '_batch_num_steps', 1),
                        restore_original_mouth=getattr(roop.globals, '_batch_restore_mouth', False),
                        selected_index=getattr(roop.globals, '_batch_selected_index', 0),
                        use_3d_recon=getattr(roop.globals, '_batch_use_3d_recon', False),
                        use_source_bank=getattr(roop.globals, '_batch_use_source_bank', False),
                        use_frontalization=getattr(roop.globals, '_batch_use_frontalization', False),
                        frontalization_threshold=getattr(roop.globals, '_batch_front_threshold', 25.0),
                        swap_model=getattr(roop.globals, '_batch_swap_model', 'inswapper'),
                    )

                if roop.globals.wait_after_extraction and temp_frame_paths:
                    extract_path = os.path.dirname(temp_frame_paths[0])
                    util.open_folder(extract_path)
                    input("Press any key to continue...")
                    print("Resorting frames to create video")
                    util.sort_rename_frames(extract_path)                                    
                
                ffmpeg.create_video(v.filename, v.finalname, fps)
                if roop.globals.keep_frames:
                    util.move_frames_to_output(v.filename, fps=fps)
                else:
                    util.delete_temp_frames(temp_frame_paths[0])
                    # If we saved originals only for per-frame mask re-processing (not keep_frames),
                    # clean them up now that the video has been compiled.
                    if per_frame_masks and not roop.globals.keep_frames:
                        orig_dir = util.get_frames_orig_path(v.filename)
                        if os.path.isdir(orig_dir):
                            import shutil as _shutil
                            _shutil.rmtree(orig_dir, ignore_errors=True)
            else:
                if util.has_extension(v.filename, ['gif']) or util.is_animated_webp(v.filename):
                    skip_audio = True
                else:
                    skip_audio = roop.globals.skip_audio
                process_mgr.run_batch_inmem(output_method, v.filename, v.finalname, v.startframe, v.endframe, fps,roop.globals.execution_threads, skip_audio)
                
            if not roop.globals.processing:
                end_processing('Processing stopped!')
                return
            
            video_file_name = v.finalname
            if os.path.isfile(video_file_name):
                destination = ''
                if util.has_extension(v.filename, ['gif']) or util.is_animated_webp(v.filename):
                    gifname = util.get_destfilename_from_path(v.filename, roop.globals.output_path, '.gif')
                    destination = util.replace_template(gifname, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    update_status('Creating final GIF')
                    # Pass fps explicitly so the GIF matches the original source
                    # timing — avoids a lossy re-detect from the intermediate MP4.
                    ffmpeg.create_gif_from_video(video_file_name, destination, target_fps=fps)
                    if os.path.isfile(destination):
                        os.remove(video_file_name)
                else:
                    skip_audio = roop.globals.skip_audio
                    destination = util.replace_template(video_file_name, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    if not skip_audio:
                        ffmpeg.restore_audio(video_file_name, v.filename, v.startframe, v.endframe, destination)
                        if os.path.isfile(destination):
                            os.remove(video_file_name)
                    else:
                        shutil.move(video_file_name, destination)

            elif is_streaming_only == False:
                update_status(f'Failed processing {os.path.basename(v.finalname)}!')
            elapsed_time = time() - start_processing
            average_fps = (v.endframe - v.startframe) / elapsed_time
            update_status(f'\nProcessing {os.path.basename(destination)} took {elapsed_time:.2f} secs, {average_fps:.2f} frames/s')
            import gc
            gc.collect()
            try:
                if torch.cuda.is_available():
                    with torch.cuda.device(roop.globals.cuda_device_id):
                        torch.cuda.empty_cache()
            except Exception:
                pass
    end_processing('Finished')


def end_processing(msg:str):
    update_status(msg)
    roop.globals.target_folder_path = None
    release_resources()


def destroy() -> None:
    if roop.globals.target_path:
        util.clean_temp(roop.globals.target_path)
    release_resources()        
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    roop.globals.CFG = Settings('config.yaml')
    roop.globals.cuda_device_id = roop.globals.startup_args.cuda_device_id
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    if roop.globals.startup_args.server_share:
        roop.globals.CFG.server_share = True
    main.run()