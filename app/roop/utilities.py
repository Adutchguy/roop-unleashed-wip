import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import sys
import urllib
import torch
import gradio
import tempfile
import cv2
import zipfile
import traceback

from pathlib import Path
from typing import List, Any
from tqdm import tqdm
from scipy.spatial import distance
from datetime import datetime

import roop.template_parser as template_parser

import roop.globals

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"

# monkey patch ssl for mac
if platform.system().lower() == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


# https://github.com/facefusion/facefusion/blob/master/facefusion
def detect_fps(target_path: str) -> float:
    # Animated WebP: OpenCV returns 0 FPS — derive from PIL frame durations instead
    if target_path and target_path.lower().endswith('.webp'):
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                n = getattr(img, 'n_frames', 1)
                if n > 1:
                    durations = []
                    for i in range(n):
                        img.seek(i)
                        d = img.info.get('duration', 40)  # duration in ms, default 40ms = 25fps
                        durations.append(d)
                    avg_ms = sum(durations) / len(durations) if durations else 40
                    return round(1000.0 / avg_ms, 2) if avg_ms > 0 else 25.0
        except Exception:
            pass
        return 25.0
    fps = 24.0
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def detect_dimensions(target_path: str):
    """Returns (width, height) for images and videos. Returns (0, 0) on failure."""
    if is_image(target_path):
        img = cv2.imread(target_path)
        if img is not None:
            return img.shape[1], img.shape[0]
        return 0, 0
    # Animated WebP: OpenCV VideoCapture returns 0x0 — use PIL instead
    if target_path and target_path.lower().endswith('.webp') and is_animated_webp(target_path):
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                return img.width, img.height
        except Exception:
            return 0, 0
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h
    cap.release()
    return 0, 0


# Gradio wants Images in RGB
def convert_to_gradio(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def sort_filenames_ignore_path(filenames):
    """Sorts a list of filenames containing a complete path by their filename,
    while retaining their original path.

    Args:
      filenames: A list of filenames containing a complete path.

    Returns:
      A sorted list of filenames containing a complete path.
    """
    filename_path_tuples = [
        (os.path.split(filename)[1], filename) for filename in filenames
    ]
    sorted_filename_path_tuples = sorted(filename_path_tuples, key=lambda x: x[0])
    return [
        filename_path_tuple[1] for filename_path_tuple in sorted_filename_path_tuples
    ]


def sort_rename_frames(path: str):
    filenames = os.listdir(path)
    filenames.sort()
    for i in range(len(filenames)):
        of = os.path.join(path, filenames[i])
        newidx = i + 1
        new_filename = os.path.join(
            path, f"{newidx:06d}." + roop.globals.CFG.output_image_format
        )
        os.rename(of, new_filename)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob(
        (
            os.path.join(
                glob.escape(temp_directory_path),
                f"*.{roop.globals.CFG.output_image_format}",
            )
        )
    )


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(
                output_path, source_name + "-" + target_name + target_extension
            )
    return output_path


def get_destfilename_from_path(
    srcfilepath: str, destfilepath: str, extension: str
) -> str:
    fn, ext = os.path.splitext(os.path.basename(srcfilepath))
    if "." in extension:
        return os.path.join(destfilepath, f"{fn}{extension}")
    return os.path.join(destfilepath, f"{fn}{extension}{ext}")


def replace_template(file_path: str, index: int = 0) -> str:
    fn, ext = os.path.splitext(os.path.basename(file_path))

    # Remove the "__temp" placeholder that was used as a temporary filename
    fn = fn.replace("__temp", "")

    template = roop.globals.CFG.output_template
    replaced_filename = template_parser.parse(
        template, {"index": str(index), "file": fn, "timestamp": datetime.now().strftime('%Y%m%d%H%M%S')}
    )

    return os.path.join(roop.globals.output_path, f"{replaced_filename}{ext}")


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def delete_temp_frames(filename: str) -> None:
    dir = os.path.dirname(os.path.dirname(filename))
    shutil.rmtree(dir)


def get_frames_output_path(target_path: str) -> str:
    """Return the directory where extracted frames are saved when keep_frames is enabled.
    Frames are placed in a <videoname>_frames sub-folder inside the configured output directory."""
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    return os.path.join(roop.globals.output_path, f"{target_name}_frames")


def move_frames_to_output(target_path: str) -> None:
    """Move the extracted temp frames to a persistent sub-folder in the output directory."""
    temp_dir = get_temp_directory_path(target_path)
    frames_out_dir = get_frames_output_path(target_path)
    if not os.path.isdir(temp_dir):
        return
    # Remove any stale frames folder from a previous run before moving
    if os.path.isdir(frames_out_dir):
        shutil.rmtree(frames_out_dir)
    shutil.move(temp_dir, frames_out_dir)
    # Clean up the now-empty parent temp directory if nothing else uses it
    parent = os.path.dirname(temp_dir)
    if os.path.exists(parent) and not os.listdir(parent):
        os.rmdir(parent)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(("png", "jpg", "jpeg", "webp"))


def has_extension(filepath: str, extensions: List[str]) -> bool:
    return filepath.lower().endswith(tuple(extensions))


def is_animated_webp(image_path: str) -> bool:
    """Return True if the file is an animated (multi-frame) WebP."""
    if not image_path or not image_path.lower().endswith(".webp"):
        return False
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return getattr(img, "n_frames", 1) > 1
    except Exception:
        return False


def is_animated_gif(image_path: str) -> bool:
    """Return True if the file is an animated (multi-frame) GIF."""
    if not image_path or not image_path.lower().endswith(".gif"):
        return False
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return getattr(img, "n_frames", 1) > 1
    except Exception:
        return False


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        if image_path.lower().endswith(".webp"):
            # Animated webp is not a static image
            return not is_animated_webp(image_path)
        if image_path.lower().endswith(".gif"):
            # Animated gif is not a static image
            return not is_animated_gif(image_path)
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(
            download_directory_path, os.path.basename(url)
        )
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get("Content-Length", 0))
            with tqdm(
                total=total,
                desc=f"Downloading {url}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


def get_local_files_from_folder(folder: str) -> List[str]:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return None
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]
    return files


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def get_device() -> str:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    print("Available ONNX providers in get_device:", available_providers)  # Debug print

    if len(roop.globals.execution_providers) < 1:
        if 'CUDAExecutionProvider' in available_providers:
            roop.globals.execution_providers = ['CUDAExecutionProvider']
            print("Forcing CUDAExecutionProvider!")  # Debug
        else:
            roop.globals.execution_providers = ["CPUExecutionProvider"]
            print("No GPU providers available—defaulting to CPU.")  # Debug

    prov = roop.globals.execution_providers[0]
    if "CoreMLExecutionProvider" in prov:
        return "mps"
    if "CUDAExecutionProvider" in prov or "ROCMExecutionProvider" in prov or "TensorrtExecutionProvider" in prov:
        print("Using GPU (cuda/rocm/tensorrt) for acceleration!")  # Debug
        return "cuda"
    if "OpenVINOExecutionProvider" in prov:
        return "mkl"
    print("Falling back to CPU.")  # Debug
    return "cpu"


def str_to_class(module_name, class_name) -> Any:
    from importlib import import_module

    class_ = None
    try:
        module_ = import_module(module_name)
        try:
            class_ = getattr(module_, class_name)()
        except AttributeError:
            print(f"Class {class_name} does not exist")
    except ImportError:
        print(f"Module {module_name} does not exist")
    return class_

def is_installed(name:str) -> bool:
    return shutil.which(name);

# Taken from https://stackoverflow.com/a/68842705
def get_platform() -> str:
    if sys.platform == "linux":
        try:
            proc_version = open("/proc/version").read()
            if "Microsoft" in proc_version:
                return "wsl"
        except:
            pass
    return sys.platform

def open_with_default_app(filename:str):
    if filename == None:
        return
    platform = get_platform()
    if platform == "darwin":
        subprocess.call(("open", filename))
    elif platform in ["win64", "win32"]:        os.startfile(filename.replace("/", "\\"))
    elif platform == "wsl":
        subprocess.call("cmd.exe /C start".split() + [filename])
    else:  # linux variants
        subprocess.call("xdg-open", filename)


def prepare_for_batch(target_files) -> str:
    print("Preparing temp files")
    tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
    if os.path.exists(tempfolder):
        shutil.rmtree(tempfolder)
    Path(tempfolder).mkdir(parents=True, exist_ok=True)
    for f in target_files:
        newname = os.path.basename(f.name)
        shutil.move(f.name, os.path.join(tempfolder, newname))
    return tempfolder


def zip(files, zipname):
    with zipfile.ZipFile(zipname, "w") as zip_file:
        for f in files:
            zip_file.write(f, os.path.basename(f))


def unzip(zipfilename: str, target_path: str):
    with zipfile.ZipFile(zipfilename, "r") as zip_file:
        zip_file.extractall(target_path)


def mkdir_with_umask(directory):
    oldmask = os.umask(0)
    # mode needs octal
    os.makedirs(directory, mode=0o775, exist_ok=True)
    os.umask(oldmask)


def open_folder(path: str):
    platform = get_platform()
    try:
        if platform == "darwin":
            subprocess.call(("open", path))
        elif platform in ["win64", "win32"]:
            open_with_default_app(path)
        elif platform == "wsl":
            subprocess.call("cmd.exe /C start".split() + [path])
        else:  # linux variants
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        traceback.print_exc()
        pass
        # import webbrowser
        # webbrowser.open(url)


def create_version_html() -> str:
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    versions_html = f"""
python: <span title="{sys.version}">{python_version}</span>
•
torch: {getattr(torch, '__long_version__',torch.__version__)}
•
gradio: {gradio.__version__}
"""
    return versions_html


def compute_cosine_distance(emb1, emb2) -> float:
    return distance.cosine(emb1, emb2)

def has_cuda_device():
    return torch.cuda is not None and torch.cuda.is_available()


def print_cuda_info():
    try:
        print(f'Number of CUDA devices: {torch.cuda.device_count()} Currently used Id: {torch.cuda.current_device()} Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    except:
       print('No CUDA device found!')

print_cuda_info()