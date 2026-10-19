import threading
from typing import Any
import insightface

import roop.globals
from roop.typing import Frame, Face

import cv2
import numpy as np
from skimage import transform as trans
from roop.capturer import get_video_frame
from roop.utilities import resolve_relative_path, conditional_download

FACE_ANALYSER = None
THREAD_LOCK_ANALYSER = threading.Lock()
THREAD_LOCK_SWAPPER = threading.Lock()
FACE_SWAPPER = None
_current_det_thresh = None


def release_face_analyser():
    global FACE_ANALYSER
    with THREAD_LOCK_ANALYSER:
        if FACE_ANALYSER is not None:
            del FACE_ANALYSER
            FACE_ANALYSER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER, _current_det_thresh

    with THREAD_LOCK_ANALYSER:
        desired_thresh = getattr(roop.globals, 'det_score_threshold', 0.5)
        if (FACE_ANALYSER is None
                or roop.globals.g_current_face_analysis != roop.globals.g_desired_face_analysis
                or _current_det_thresh != desired_thresh):
            model_path = resolve_relative_path('..')
            allowed_modules = roop.globals.g_desired_face_analysis
            roop.globals.g_current_face_analysis = roop.globals.g_desired_face_analysis
            _current_det_thresh = desired_thresh
            if roop.globals.CFG.force_cpu:
                print("Forcing CPU for Face Analysis")
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    root=model_path, providers=["CPUExecutionProvider"], allowed_modules=allowed_modules
                )
            else:
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name="buffalo_l", root=model_path, providers=roop.globals.execution_providers, allowed_modules=allowed_modules
                )
            FACE_ANALYSER.prepare(
                ctx_id=0,
                det_size=(640, 640) if roop.globals.default_det_size else (320, 320),
                det_thresh=desired_thresh,
            )
    return FACE_ANALYSER


def get_first_face(frame: Frame) -> Any:
    try:
        faces = get_face_analyser().get(frame)
        return min(faces, key=lambda x: x.bbox[0])
    #   return sorted(faces, reverse=True, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[0]
    except:
        return None


def get_all_faces(frame: Frame) -> Any:
    try:
        faces = get_face_analyser().get(frame)
        return sorted(faces, key=lambda x: x.bbox[0])
    except:
        return None


# ---------------------------------------------------------------------------
# Orientation-robust detection
#
# The face detector (and the landmark / recognition models downstream of it)
# are NOT rotation invariant.  A face lying on its side or upside down is
# detected with low confidence, with garbage landmarks, or not at all -- so
# anything that only "corrects" rotation AFTER a plain detection (the old
# behaviour) is working from the worst possible measurements.  Manually
# rotating the media first works so much better because the detector then
# sees an upright face.
#
# get_all_faces_any_orientation() reproduces that automatically: it runs the
# (cheap) detector on the frame rotated by 0/90/180/270 degrees, keeps the
# view in which each face looks most upright, runs the expensive models
# (landmarks, embedding, ...) only on that upright view, and maps every
# coordinate back into the ORIGINAL frame.  Any face is therefore within 45
# degrees of upright in at least one view, whatever angle it is at and
# whatever it did on the previous frame (no temporal state is needed).
# ---------------------------------------------------------------------------
_ORIENT_EXTRA_MIN_SCORE = 0.6    # stricter confidence for the 90/180/270 views
_ORIENT_MAX_VIEW_ROLL = 70.0     # reject 90/180/270 hits that still look sideways
_ORIENT_ROLL_PENALTY = 0.3       # quality = det_score - penalty * |roll|/90
_ORIENT_NMS_IOU = 0.3


def rotate_image_k90(image, k):
    """Lossless rotation by k * 90 degrees counter-clockwise (pure pixel moves)."""
    k = int(k) % 4
    if k == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, k))


def _unrotate_points_k90(pts, k, orig_w, orig_h):
    """Map points (N, >=2) from an image that was rotated by k * 90 degrees CCW
    back into the coordinate space of the original (orig_w x orig_h) image.
    Extra columns (e.g. the z of 3D landmarks) are passed through untouched."""
    out = np.array(pts, dtype=np.float32, copy=True)
    k = int(k) % 4
    if k == 0:
        return out
    x = out[:, 0].copy()
    y = out[:, 1].copy()
    if k == 1:
        out[:, 0] = orig_w - 1 - y
        out[:, 1] = x
    elif k == 2:
        out[:, 0] = orig_w - 1 - x
        out[:, 1] = orig_h - 1 - y
    else:
        out[:, 0] = y
        out[:, 1] = orig_h - 1 - x
    return out


def _unrotate_bbox_k90(bbox, k, orig_w, orig_h):
    pts = _unrotate_points_k90(np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]]), k, orig_w, orig_h)
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)


def _unrotate_face_k90(face, k, orig_w, orig_h):
    """In-place: express every spatial attribute of `face` in original-frame
    coordinates.  Embedding / gender / age are rotation independent."""
    face.bbox = _unrotate_bbox_k90(face.bbox, k, orig_w, orig_h)
    if face.kps is not None:
        face.kps = _unrotate_points_k90(face.kps, k, orig_w, orig_h)
    if getattr(face, 'landmark_2d_106', None) is not None:
        face.landmark_2d_106 = _unrotate_points_k90(face.landmark_2d_106, k, orig_w, orig_h)
    if getattr(face, 'landmark_3d_68', None) is not None:
        face.landmark_3d_68 = _unrotate_points_k90(face.landmark_3d_68, k, orig_w, orig_h)
    return face


def _boxes_overlap(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return False
    inter = iw * ih
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    iou = inter / (area_a + area_b - inter)
    return iou > _ORIENT_NMS_IOU or inter / min(area_a, area_b) > 0.6


def get_all_faces_any_orientation(frame: Frame) -> Any:
    """Like get_all_faces(), but finds faces at ANY in-plane angle.

    Returned Face objects are in original-frame coordinates (bbox, kps,
    landmark_2d_106, landmark_3d_68) but their landmarks / embedding were
    computed on the view in which the face is closest to upright, so they are
    as accurate as if the media had been rotated by hand beforehand.
    """
    try:
        if frame is None:
            return None
        analyser = get_face_analyser()
        det_model = getattr(analyser, 'det_model', None)
        if det_model is None:
            return get_all_faces(frame)

        from insightface.app.common import Face as InsightFace

        h, w = frame.shape[:2]
        det_thresh = float(getattr(roop.globals, 'det_score_threshold', 0.5))
        views = {}
        cands = []
        for k in range(4):
            img = rotate_image_k90(frame, k)
            views[k] = img
            bboxes, kpss = det_model.detect(img, max_num=0, metric='default')
            if bboxes is None or bboxes.shape[0] == 0:
                continue
            for i in range(bboxes.shape[0]):
                score = float(bboxes[i, 4])
                kps_r = None if kpss is None else np.array(kpss[i], dtype=np.float32)
                roll = 0.0
                if kps_r is not None:
                    roll = estimate_roll_angle(kps_r)
                if k != 0:
                    # Sideways views are only trusted when the face really
                    # looks upright in them and the detector is confident.
                    if kps_r is None or score < max(det_thresh, _ORIENT_EXTRA_MIN_SCORE):
                        continue
                    if abs(roll) > _ORIENT_MAX_VIEW_ROLL:
                        continue
                quality = score - _ORIENT_ROLL_PENALTY * min(abs(roll), 90.0) / 90.0
                cands.append({
                    'k': k,
                    'score': score,
                    'quality': quality,
                    'bbox_r': np.array(bboxes[i, 0:4], dtype=np.float32),
                    'bbox': _unrotate_bbox_k90(bboxes[i, 0:4], k, w, h),
                    'kps_r': kps_r,
                })

        cands.sort(key=lambda c: c['quality'], reverse=True)
        kept = []
        for c in cands:
            if any(_boxes_overlap(c['bbox'], kc['bbox']) for kc in kept):
                continue
            kept.append(c)

        faces = []
        for c in kept:
            k = c['k']
            face = InsightFace(bbox=c['bbox_r'], kps=c['kps_r'], det_score=c['score'])
            for taskname, model in analyser.models.items():
                if taskname == 'detection':
                    continue
                model.get(views[k], face)
            if k:
                _unrotate_face_k90(face, k, w, h)
            faces.append(face)
        views.clear()
        return sorted(faces, key=lambda x: x.bbox[0])
    except Exception:
        return None


def get_first_face_any_orientation(frame: Frame) -> Any:
    try:
        faces = get_all_faces_any_orientation(frame)
        return min(faces, key=lambda x: x.bbox[0])
    except Exception:
        return None


def get_all_faces_auto(frame: Frame) -> Any:
    """get_all_faces(), upgraded to any-orientation detection when the
    'Auto rotate tilted Faces' option is on."""
    if getattr(roop.globals, 'autorotate_faces', False):
        return get_all_faces_any_orientation(frame)
    return get_all_faces(frame)


def get_first_face_auto(frame: Frame) -> Any:
    if getattr(roop.globals, 'autorotate_faces', False):
        return get_first_face_any_orientation(frame)
    return get_first_face(frame)


def extract_face_images(source_filename, video_info, extra_padding=-1.0):
    face_data = []
    source_image = None

    if video_info[0]:
        frame = get_video_frame(source_filename, video_info[1])
        if frame is not None:
            source_image = frame
        else:
            return face_data
    else:
        source_image = cv2.imdecode(np.fromfile(source_filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    faces = get_all_faces(source_image)
    if faces is None:
        return face_data

    i = 0
    for face in faces:
        (startX, startY, endX, endY) = face["bbox"].astype("int")
        startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
        if extra_padding > 0.0:
            if source_image.shape[:2] == (512, 512):
                i += 1
                face_data.append([face, source_image])
                continue

            found = False
            for i in range(1, 3):
                (startX, startY, endX, endY) = face["bbox"].astype("int")
                startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
                cutout_padding = extra_padding
                # top needs extra room for detection
                padding = int((endY - startY) * cutout_padding)
                oldY = startY
                startY -= padding

                factor = 0.25 if i == 1 else 0.5
                cutout_padding = factor
                padding = int((endY - oldY) * cutout_padding)
                endY += padding
                padding = int((endX - startX) * cutout_padding)
                startX -= padding
                endX += padding
                startX, endX, startY, endY = clamp_cut_values(
                    startX, endX, startY, endY, source_image
                )
                face_temp = source_image[startY:endY, startX:endX]
                face_temp = resize_image_keep_content(face_temp)
                testfaces = get_all_faces(face_temp)
                if testfaces is not None and len(testfaces) > 0:
                    i += 1
                    face_data.append([testfaces[0], face_temp])
                    found = True
                    break

            if not found:
                print("No face found after resizing, this shouldn't happen!")
            continue

        face_temp = source_image[startY:endY, startX:endX]
        if face_temp.size < 1:
            continue

        i += 1
        face_data.append([face, face_temp])
    return face_data


def clamp_cut_values(startX, endX, startY, endY, image):
    if startX < 0:
        startX = 0
    if endX > image.shape[1]:
        endX = image.shape[1]
    if startY < 0:
        startY = 0
    if endY > image.shape[0]:
        endY = image.shape[0]
    return startX, endX, startY, endY



def face_offset_top(face: Face, offset):
    face["bbox"][1] += offset
    face["bbox"][3] += offset
    lm106 = face.landmark_2d_106
    add = np.full_like(lm106, [0, offset])
    face["landmark_2d_106"] = lm106 + add
    return face


def resize_image_keep_content(image, new_width=512, new_height=512):
    dim = None
    (h, w) = image.shape[:2]
    if h > w:
        r = new_height / float(h)
        dim = (int(w * r), new_height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = new_width / float(w)
        dim = (new_width, int(h * r))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    (h, w) = image.shape[:2]
    if h == new_height and w == new_width:
        return image
    resize_img = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    offs = (new_width - w) if h == new_height else (new_height - h)
    startoffs = int(offs // 2) if offs % 2 == 0 else int(offs // 2) + 1
    offs = int(offs // 2)

    if h == new_height:
        resize_img[0:new_height, startoffs : new_width - offs] = image
    else:
        resize_img[startoffs : new_height - offs, 0:new_width] = image
    return resize_img


def rotate_image_90(image, rotate=True):
    if rotate:
        return np.rot90(image)
    else:
        return np.rot90(image, 1, (1, 0))


def rotate_anticlockwise(frame):
    return rotate_image_90(frame)


def rotate_clockwise(frame):
    return rotate_image_90(frame, False)


def rotate_image_180(image):
    return np.flip(image, 0)


def rotate_image_same_size(image, angle_degrees):
    """Rotate `image` by an arbitrary angle (degrees, CCW positive, matching
    cv2.getRotationMatrix2D) around its own center, keeping the same
    width/height as the input.

    Content that rotates outside the original frame is lost (filled black).
    That's fine for a cheap trial detection during angle search -- only the
    central face region needs to stay intact -- but NOT for the frame that
    actually gets swapped/pasted back: use rotate_image_any + unrotate_image_any
    for that, which never introduce black by clipping corners.
    """
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_degrees, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=0.0)


def _snap_quarter_turns(angle_degrees, tol=0.5):
    """k (0..3) if `angle_degrees` is within `tol` of a multiple of 90, else None."""
    r = int(round(angle_degrees / 90.0))
    if abs(angle_degrees - r * 90.0) <= tol:
        return r % 4
    return None


def _quarter_turn_matrix(k, w, h):
    """Forward 2x3 affine (cv2 convention, pixel-index coordinates) equivalent
    to rotate_image_k90(image, k) for an input of size (w, h)."""
    if k == 1:
        return np.array([[0, 1, 0], [-1, 0, w - 1]], dtype=np.float64)
    if k == 2:
        return np.array([[-1, 0, w - 1], [0, -1, h - 1]], dtype=np.float64)
    if k == 3:
        return np.array([[0, -1, h - 1], [1, 0, 0]], dtype=np.float64)
    return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)


def rotate_image_any(image, angle_degrees):
    """Rotate `image` by an arbitrary angle (degrees, CCW positive) around its
    own center, expanding the canvas so the ENTIRE rotated image is kept --
    no source content is clipped/blackened (unlike rotate_image_same_size).

    Angles within 0.5 degrees of a multiple of 90 are executed as an exact,
    lossless pixel rotation (np.rot90) -- identical to rotating the media by
    hand in an editor -- instead of an interpolated warp.

    Returns (rotated_image, M, (orig_w, orig_h)). M is the forward affine
    matrix and (orig_w, orig_h) is the input size; pass both to
    unrotate_image_any to map a result computed in the rotated canvas back
    to the input's exact coordinate frame via the true affine inverse --
    NOT a second forward-style rotation, which would re-clip corners and
    scatter black wedges across the reconstructed image.
    """
    (h, w) = image.shape[:2]
    k = _snap_quarter_turns(angle_degrees)
    if k is not None:
        rotated = image.copy() if k == 0 else rotate_image_k90(image, k)
        return rotated, _quarter_turn_matrix(k, w, h), (w, h)
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_degrees, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(np.ceil(round(h * sin + w * cos, 6)))
    new_h = int(np.ceil(round(h * cos + w * sin, 6)))
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=0.0)
    return rotated, M, (w, h)


def unrotate_image_any(image, M, original_size):
    """Exact inverse of rotate_image_any.

    Maps `image` (same shape as the expanded canvas rotate_image_any
    produced) back to the original (width, height) frame using the true
    affine inverse of M, so every pixel within the reconstructed rectangle
    comes from real source content -- no black wedge is introduced, unlike
    applying a second forward rotation.
    """
    w, h = original_size
    M_inv = cv2.invertAffineTransform(M)
    return cv2.warpAffine(image, M_inv, (w, h), borderValue=0.0)


def estimate_roll_angle(kps):
    """Signed in-plane roll of a face, in degrees (-180..180], from its
    5-point keypoints.  0 = upright; positive = the face is rolled clockwise
    on screen, i.e. rotating the image by that many degrees counter-clockwise
    (what cv2.getRotationMatrix2D does for positive angles) makes it upright.

    Two independent cues are combined so one noisy / hidden eye cannot
    dominate: the eye-to-eye line (kps[0] -> kps[1]; kps[0] is the image-left
    eye of an upright face) and the face's own vertical axis (mouth centre ->
    eye centre).  Valid over the full circle, not just +-90 degrees.
    """
    k = np.asarray(kps, dtype=np.float64)
    ex = k[1][0] - k[0][0]
    ey = k[1][1] - k[0][1]
    eye_angle = np.arctan2(ey, ex)
    if len(k) >= 5:
        ux = (k[0][0] + k[1][0]) / 2.0 - (k[3][0] + k[4][0]) / 2.0
        uy = (k[0][1] + k[1][1]) / 2.0 - (k[3][1] + k[4][1]) / 2.0
        if np.hypot(ex, ey) > 1e-6 and np.hypot(ux, uy) > 1e-6:
            axis_angle = np.arctan2(ux, -uy)
            vx = np.cos(eye_angle) + np.cos(axis_angle)
            vy = np.sin(eye_angle) + np.sin(axis_angle)
            if np.hypot(vx, vy) > 1e-6:
                return float(np.degrees(np.arctan2(vy, vx)))
    return float(np.degrees(eye_angle))


_ANCHOR_MAX_REL_DIST = 0.75   # target face centre must be within this many face-sizes of the anchor


def _select_index(bboxes, anchor_xy):
    """Index of the detection to use: the one nearest `anchor_xy` (and close
    enough to plausibly BE the anchored face), or -- with no anchor -- the
    leftmost one, matching get_first_face()."""
    n = bboxes.shape[0]
    if n == 0:
        return None
    if anchor_xy is None:
        return int(np.argmin(bboxes[:, 0]))
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    d = np.hypot(cx - anchor_xy[0], cy - anchor_xy[1])
    i = int(np.argmin(d))
    size = max(bboxes[i, 2] - bboxes[i, 0], bboxes[i, 3] - bboxes[i, 1])
    return i if d[i] <= size * _ANCHOR_MAX_REL_DIST else None


def _pick_face(get_first_face_fn, img, anchor_xy=None):
    """Full analysis of `img`, returning the face nearest `anchor_xy` (so a
    neighbour that happens to be further left in a group shot is never
    mistaken for the face being straightened), or get_first_face_fn(img) when
    no anchor is given."""
    if anchor_xy is None:
        return get_first_face_fn(img)
    faces = get_all_faces(img)
    if not faces:
        return None
    i = _select_index(np.array([f.bbox for f in faces], dtype=np.float32), anchor_xy)
    return None if i is None else faces[i]


def _trial_kps(get_first_face_fn, img, anchor_xy):
    """Keypoints of the anchored face in `img`.  With the stock detector this
    runs ONLY the detection network (no landmark / recognition models), which
    is all an angle trial needs."""
    if get_first_face_fn is get_first_face:
        try:
            bboxes, kpss = get_face_analyser().det_model.detect(img, max_num=0, metric='default')
        except Exception:
            return None
        if bboxes is None or kpss is None or bboxes.shape[0] == 0:
            return None
        i = _select_index(bboxes, anchor_xy)
        return None if i is None else kpss[i]
    face = _pick_face(get_first_face_fn, img, anchor_xy)
    if face is None or getattr(face, 'kps', None) is None:
        return None
    return face.kps


def find_upright_rotation(get_first_face_fn, cutframe, rough_angle, threshold=5.0, anchor=None, max_refinements=3):
    """Rotate a padded face cutout so the face is upright, re-detecting after
    each trial rotation so downstream alignment/swap gets accurate landmarks
    instead of ones estimated from a tilted view.

    `rough_angle` is the face's roll (degrees) measured on the un-rotated
    frame.  It is tried first; if the re-detected face is still more than
    `threshold` degrees off, the opposite sign is tried too (so a sign
    mistake can never make things worse than doing nothing) and the angle is
    then refined from the measured residual, up to `max_refinements` rounds.
    Every trial is a real, lossless-canvas rotation (rotate_image_any, which
    is an exact pixel rotation for multiples of 90 degrees) of the same
    pixels that are returned, so the winning trial is reused as-is.

    `anchor` is the target face's centre in `cutframe` coordinates.  When
    given, the face nearest it is used in every re-detection rather than the
    leftmost face of the cutout -- essential in group shots where a
    neighbour's face falls inside the padded crop.

    Returns (rotated_frame, face, M, original_size, total_angle_applied), or
    (None, None, None, None, 0.0) if the face is already upright, if no
    candidate rotation yields a usable detection, or if rotating would not
    beat the unrotated roll (caller should fall back to the un-rotated frame).
    """
    if rough_angle is None or abs(rough_angle) < threshold:
        return None, None, None, None, 0.0

    def anchor_in(M):
        if anchor is None:
            return None
        return (M[0, 0] * anchor[0] + M[0, 1] * anchor[1] + M[0, 2],
                M[1, 0] * anchor[0] + M[1, 1] * anchor[1] + M[1, 2])

    cache = {}

    def trial(angle):
        angle = ((float(angle) + 180.0) % 360.0) - 180.0
        key = round(angle, 1)
        if key not in cache:
            rotated, M, size = rotate_image_any(cutframe, angle)
            kps = _trial_kps(get_first_face_fn, rotated, anchor_in(M))
            residual = None if kps is None else estimate_roll_angle(kps)
            cache[key] = (angle, residual, rotated, M, size)
        return cache[key]

    best = [None]   # (abs_residual, angle, residual, key)

    def consider(angle):
        t = trial(angle)
        if t[1] is None:
            return
        if best[0] is None or abs(t[1]) < best[0][0]:
            best[0] = (abs(t[1]), t[0], t[1], round(t[0], 1))

    consider(rough_angle)
    if best[0] is None or best[0][0] >= threshold:
        consider(-rough_angle)
    for _ in range(max_refinements):
        if best[0] is None or best[0][0] < threshold:
            break
        before, ang, res, _k = best[0]
        consider(ang + res)
        consider(ang - res)
        if best[0][0] > before - 0.5:
            break

    if best[0] is None or best[0][0] >= abs(rough_angle):
        return None, None, None, None, 0.0

    _, _, _, key = best[0]
    angle, _res, rotated, M, size = cache[key]
    face = _pick_face(get_first_face_fn, rotated, anchor_in(M))
    if face is None or getattr(face, 'kps', None) is None:
        return None, None, None, None, 0.0
    return rotated, face, M, size, angle


# alignment code from insightface https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    elif image_size % 512 == 0:
        ratio = float(image_size) / 512.0
        diff_x = 32.0 * ratio

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M



# aligned, M = norm_crop2(f[1], face.kps, 512)
def align_crop(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[: resized_im.shape[0], : resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)
    
def create_blank_image(width, height):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:] = [0,0,0,0]
    return img

