from typing import Optional
import cv2
import numpy as np

from roop.typing import Frame

current_video_path = None
current_frame_total = 0
current_capture = None

# PIL-based cache for animated webp (PIL is already in the env via clip)
_awebp_path = None
_awebp_frames = None   # list of BGR numpy arrays, populated lazily


def _load_animated_webp(video_path: str):
    """Load all frames of an animated WebP into memory as BGR arrays."""
    global _awebp_path, _awebp_frames
    if _awebp_path == video_path and _awebp_frames is not None:
        return
    from PIL import Image
    frames = []
    with Image.open(video_path) as img:
        for i in range(getattr(img, "n_frames", 1)):
            img.seek(i)
            frame_rgb = np.array(img.convert("RGB"))
            frames.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    _awebp_path = video_path
    _awebp_frames = frames


def get_image_frame(filename: str):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        print(f"Exception reading {filename}")
    return None


def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:
    global current_video_path, current_capture, current_frame_total

    # Animated WebP — use PIL-based reader
    if video_path.lower().endswith(".webp"):
        _load_animated_webp(video_path)
        if _awebp_frames:
            idx = max(0, min(len(_awebp_frames) - 1, frame_number - 1))
            return _awebp_frames[idx]
        return None

    if video_path != current_video_path:
        release_video()
        current_capture = cv2.VideoCapture(video_path)
        current_video_path = video_path
        current_frame_total = current_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    target = max(0, min(int(current_frame_total) - 1, frame_number - 1))
    current_capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    has_frame, frame = current_capture.read()
    if has_frame:
        return frame
    return None

def release_video():
    global current_capture

    if current_capture is not None:
        current_capture.release()
        current_capture = None


def get_video_frame_total(video_path: str) -> int:
    # Animated WebP — use PIL
    if video_path.lower().endswith(".webp"):
        try:
            from PIL import Image
            with Image.open(video_path) as img:
                return getattr(img, "n_frames", 1)
        except Exception:
            return 1
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
