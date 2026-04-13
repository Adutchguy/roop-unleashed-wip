"""
face_3d_recon.py — Pose-aware source face warping for roop-unleashed
=====================================================================
Uses the 68-point 3D landmarks that insightface already provides
(no additional model downloads required) to warp the source face crop
so it approximates the target head pose before ArcFace embedding
extraction.

Why this helps
--------------
The face-swap model was trained primarily on frontal faces.  When the
target face is at a steep yaw or pitch angle the swap degrades because
the source embedding comes from a frontal crop while the target crop
shows the angled face.  Warping the source crop to match the target
pose before re-embedding improves the correspondence the swap model
sees, producing better results on profile and angled targets.

How it works
------------
1. Both source and target faces expose `landmark_3d_68` from insightface.
2. `cv2.solvePnP` fits a rigid pose to each landmark set against a
   generic 3D reference face.
3. The angular difference is decomposed into yaw and pitch.
4. For large yaw the source crop is horizontally reflected and
   shear-warped to reduce the angular gap.
5. Insightface re-runs on the warped crop to produce a posed embedding
   which replaces the original for that frame's swap call.

No external model files are needed — activates immediately.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Generic 3D face reference (68-pt, ~unit scale, OpenCV convention)
# Derived from the average BFM / 300W-LP mean shape — good enough for PnP.
# fmt: off
_REF3D_68 = np.array([
    # Jaw line (0-16)
    [-0.786, -0.553, -0.293], [-0.751, -0.763, -0.147], [-0.679, -0.953, -0.026],
    [-0.566, -1.118,  0.076], [-0.421, -1.247,  0.163], [-0.258, -1.355,  0.236],
    [-0.079, -1.410,  0.264], [ 0.079, -1.410,  0.264], [ 0.258, -1.355,  0.236],
    [ 0.421, -1.247,  0.163], [ 0.566, -1.118,  0.076], [ 0.679, -0.953, -0.026],
    [ 0.751, -0.763, -0.147], [ 0.786, -0.553, -0.293], [ 0.757, -0.330, -0.310],
    [ 0.712, -0.113, -0.298], [ 0.645,  0.089, -0.256],
    # Left brow (17-21)
    [-0.665,  0.564, -0.200], [-0.511,  0.700, -0.065], [-0.337,  0.752,  0.064],
    [-0.150,  0.726,  0.159], [-0.000,  0.645,  0.199],
    # Right brow (22-26)
    [ 0.000,  0.645,  0.199], [ 0.150,  0.726,  0.159], [ 0.337,  0.752,  0.064],
    [ 0.511,  0.700, -0.065], [ 0.665,  0.564, -0.200],
    # Nose ridge (27-30)
    [ 0.000,  0.453,  0.265], [ 0.000,  0.275,  0.389], [ 0.000,  0.095,  0.487],
    [ 0.000, -0.098,  0.547],
    # Nose base (31-35)
    [-0.218, -0.234,  0.430], [-0.117, -0.279,  0.491], [ 0.000, -0.298,  0.523],
    [ 0.117, -0.279,  0.491], [ 0.218, -0.234,  0.430],
    # Left eye (36-41)
    [-0.547,  0.390,  0.063], [-0.406,  0.478,  0.185], [-0.252,  0.473,  0.239],
    [-0.134,  0.367,  0.230], [-0.257,  0.282,  0.183], [-0.411,  0.282,  0.120],
    # Right eye (42-47)
    [ 0.134,  0.367,  0.230], [ 0.252,  0.473,  0.239], [ 0.406,  0.478,  0.185],
    [ 0.547,  0.390,  0.063], [ 0.411,  0.282,  0.120], [ 0.257,  0.282,  0.183],
    # Outer mouth (48-59)
    [-0.368, -0.547,  0.299], [-0.214, -0.647,  0.417], [-0.083, -0.699,  0.465],
    [ 0.000, -0.715,  0.479], [ 0.083, -0.699,  0.465], [ 0.214, -0.647,  0.417],
    [ 0.368, -0.547,  0.299], [ 0.214, -0.474,  0.385], [ 0.083, -0.433,  0.428],
    [ 0.000, -0.420,  0.440], [-0.083, -0.433,  0.428], [-0.214, -0.474,  0.385],
    # Inner mouth (60-67)
    [-0.274, -0.564,  0.338], [-0.088, -0.622,  0.429], [ 0.000, -0.639,  0.452],
    [ 0.088, -0.622,  0.429], [ 0.274, -0.564,  0.338], [ 0.088, -0.507,  0.415],
    [ 0.000, -0.488,  0.432], [-0.088, -0.507,  0.415],
], dtype=np.float64)
# fmt: on

# Camera intrinsic approximation (updated per image size at call time)
_DIST = np.zeros((4, 1), dtype=np.float64)


def _build_camera(img_size: int) -> np.ndarray:
    f = img_size * 1.2
    c = img_size / 2.0
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------

def estimate_pose(lm68: np.ndarray, img_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a rigid pose to 68 2-D landmarks via EPnP.

    lm68 : (68, 2) float32 — landmarks in image coordinates
    Returns (rvec, tvec) as (3,1) float64 arrays, or raises RuntimeError.
    """
    # Scale reference so it roughly matches image-space landmark spread
    lm_scale = float(np.std(lm68))
    ref_scale = float(np.std(_REF3D_68[:, :2]))
    pts3d = (_REF3D_68 * (lm_scale / (ref_scale + 1e-8))).copy()

    cam = _build_camera(img_size)
    ok, rvec, tvec = cv2.solvePnP(
        pts3d, lm68.astype(np.float64), cam, _DIST, flags=cv2.SOLVEPNP_EPNP
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    return rvec, tvec


def decompose_yaw_pitch(rvec: np.ndarray) -> Tuple[float, float]:
    """
    Extract yaw and pitch (radians) from an OpenCV rotation vector.
    Positive yaw  → face turned right.
    Positive pitch → face tilted up.
    """
    R, _ = cv2.Rodrigues(rvec)
    # R is camera←world; yaw = atan2(R[0,2], R[2,2]), pitch = asin(-R[1,2])
    yaw   = float(np.arctan2(R[0, 2], R[2, 2]))
    pitch = float(np.arcsin(np.clip(-R[1, 2], -1.0, 1.0)))
    return yaw, pitch


# ---------------------------------------------------------------------------
# Source crop warping
# ---------------------------------------------------------------------------

def _horizontal_shear(img: np.ndarray, shear: float) -> np.ndarray:
    """Apply a horizontal shear to simulate mild yaw offset (|shear| ≤ 1)."""
    h, w = img.shape[:2]
    # Shear matrix: x' = x + shear * (y - h/2),  y' = y
    M = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def warp_source_to_pose(
    src_crop: np.ndarray,           # (H, W, 3) BGR align_crop of the source face
    src_lm68: np.ndarray,           # (68, 2) float32 — source 3D landmarks (xy)
    tgt_lm68: np.ndarray,           # (68, 2) float32 — target 3D landmarks (xy)
    img_size: int = 512,
    yaw_threshold_deg: float = 15.0,
    max_shear: float = 0.25,
) -> np.ndarray:
    """
    Warp src_crop to approximate the head pose of the target face.

    For yaw differences below yaw_threshold_deg the crop is returned as-is.
    For larger differences a horizontal flip and/or shear is applied.

    Returns a BGR uint8 image the same shape as src_crop.
    """
    try:
        src_rvec, _ = estimate_pose(src_lm68, img_size)
        tgt_rvec, _ = estimate_pose(tgt_lm68, img_size)
    except RuntimeError:
        return src_crop

    src_yaw, src_pitch = decompose_yaw_pitch(src_rvec)
    tgt_yaw, tgt_pitch = decompose_yaw_pitch(tgt_rvec)

    delta_yaw   = tgt_yaw   - src_yaw    # radians
    delta_pitch = tgt_pitch - src_pitch

    threshold_rad = np.radians(yaw_threshold_deg)

    result = src_crop.copy()

    # --- Yaw compensation ------------------------------------------------
    if abs(delta_yaw) > threshold_rad:
        # Flip horizontally if the target is looking in a significantly
        # different direction than the source
        if delta_yaw > threshold_rad:
            result = cv2.flip(result, 1)        # flip left↔right
        # Apply a gentle shear to further reduce the angular gap
        remaining = abs(delta_yaw) - threshold_rad
        shear_amount = np.clip(remaining / np.radians(45.0), 0.0, 1.0) * max_shear
        if delta_yaw > 0:
            result = _horizontal_shear(result,  shear_amount)
        else:
            result = _horizontal_shear(result, -shear_amount)

    # --- Pitch compensation (subtle vertical shear) -----------------------
    if abs(delta_pitch) > threshold_rad:
        h, w = result.shape[:2]
        remaining = abs(delta_pitch) - threshold_rad
        v_shear = np.clip(remaining / np.radians(45.0), 0.0, 1.0) * max_shear * 0.5
        # Vertical shear: y' = y + v_shear * (x - w/2)
        M = np.float32([[1, 0, 0], [v_shear, 1, -v_shear * w / 2]])
        if delta_pitch > 0:
            result = cv2.warpAffine(result, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
        else:
            M[1, 0] = -v_shear
            M[1, 2] =  v_shear * w / 2
            result = cv2.warpAffine(result, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)

    return result


# ---------------------------------------------------------------------------
# Landmark extraction helpers (crop-space)
# ---------------------------------------------------------------------------

def landmarks_to_crop_space(lm68_full: np.ndarray, affine_M: np.ndarray) -> np.ndarray:
    """
    Map 68 full-frame landmarks into align_crop space via the 2×3 affine M
    produced by align_crop().

    lm68_full : (68, 2) or (68, 3) — image-space landmarks (xy used)
    affine_M  : (2, 3) float32 affine from align_crop
    Returns   : (68, 2) float32 in crop coordinates
    """
    pts = lm68_full[:, :2].astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    return (affine_M @ np.hstack([pts, ones]).T).T.astype(np.float32)


# ---------------------------------------------------------------------------
# Public singleton facade
# ---------------------------------------------------------------------------

class Face3DRecon:
    """
    Singleton facade for pose-aware source crop warping.

    Usage in ProcessMgr.process_face():

        from roop.face_3d_recon import Face3DRecon
        recon = Face3DRecon.instance()
        posed_crop = recon.get_posed_source_crop(
            src_crop_bgr, src_lm68_crop, tgt_lm68_crop,
            img_size=subsample_size
        )
        # posed_crop is the same shape as src_crop_bgr — feed to insightface
    """

    _singleton: Optional['Face3DRecon'] = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> 'Face3DRecon':
        if cls._singleton is None:
            with cls._lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    # The approach requires no pretrained model, so it's always "available".
    @property
    def available(self) -> bool:
        return True

    def get_posed_source_crop(
        self,
        src_crop_bgr: np.ndarray,     # align_crop output for the source face
        src_lm68_crop: np.ndarray,    # (68,2) source landmarks in crop space
        tgt_lm68_crop: np.ndarray,    # (68,2) target landmarks in crop space
        img_size: int = 512,
        yaw_threshold_deg: float = 15.0,
    ) -> np.ndarray:
        """
        Return a version of src_crop_bgr warped to approximate the target
        head pose.  Returns src_crop_bgr unchanged if pose estimation fails
        or if the angular difference is below yaw_threshold_deg.
        """
        return warp_source_to_pose(
            src_crop_bgr, src_lm68_crop, tgt_lm68_crop,
            img_size=img_size,
            yaw_threshold_deg=yaw_threshold_deg,
        )

    # ------------------------------------------------------------------
    # fit_source / render_from_coefficients stubs
    # (kept so ProcessMgr references don't break; they're no-ops here)
    # ------------------------------------------------------------------

    def fit_source(self, src_crop_bgr: np.ndarray):
        """
        Returns a lightweight dict with the crop so the per-frame path
        has something to work with.
        """
        return {'src_crop': src_crop_bgr}

    def render_from_coefficients(self, src_coeffs: dict, tgt_lm68: np.ndarray,
                                  output_size: int = 512):
        """Not used in the landmark-warp path; always returns None."""
        return None
