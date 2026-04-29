"""
face_frontalize.py — Affine-based face crop frontalization for roop-unleashed
==============================================================================
Warps a canonical align_crop toward a frontal pose using a global least-squares
affine estimated from 68-point 3D landmarks.

Why this helps profile swaps
-----------------------------
The swap model (inswapper / GHOST) was trained on near-frontal faces.  When the
target crop is at a steep yaw or pitch angle the model generates poor output
because the crop geometry doesn't match its training distribution.  Frontalizing
the crop before the swap gives the model a near-frontal input, producing better
results; the inverse affine then maps the swapped result back to the original pose
before blending it into the frame.

Algorithm
---------
1.  Fit the current face pose via solvePnP using the 68-pt crop-space landmarks.
2.  Re-project the same 3D reference points with **zero rotation** (frontal pose)
    but the same translation → gives "frontal" 2-D positions in crop space.
3.  Fit a full 2×3 affine  M  using ``cv2.estimateAffine2D(actual_lm68, frontal_lm68)``.
4.  ``cv2.warpAffine(crop, M)``  produces the frontalized crop.
5.  After the swap,  ``cv2.invertAffineTransform(M)``  maps the result back.

This approach is self-consistent: the frontal reference is derived from the same
PnP camera model used to compute the pose angles, so the target landmark positions
are always compatible with the actual landmark coordinate space.

No extra model downloads — uses the same ``landmark_3d_68`` from buffalo_l.
"""

from __future__ import annotations

import math
import cv2
import numpy as np
from typing import Optional, Tuple

from roop.face_3d_recon import _REF3D_68, _DIST, _build_camera


# ---------------------------------------------------------------------------
# Pose-based frontal landmark computation
# ---------------------------------------------------------------------------

def get_frontal_landmarks_from_pose(
    lm68_crop: np.ndarray,   # (68, 2) float32 — landmarks in crop space
    crop_size: int,
) -> Optional[np.ndarray]:
    """
    Compute where the 68 landmarks would appear if the face were exactly frontal,
    by fitting the current pose via solvePnP and re-projecting with zero rotation.

    This is coordinate-system–consistent: both the actual and frontal positions
    are in the same crop-space coordinate frame, so estimateAffine2D works
    correctly.

    Returns (68, 2) float32 in crop coordinates, or None if PnP fails.
    """
    # Scale reference points to match landmark spread (mirrors estimate_pose)
    lm_scale = float(np.std(lm68_crop))
    ref_scale = float(np.std(_REF3D_68[:, :2]))
    pts3d = (_REF3D_68 * (lm_scale / max(ref_scale, 1e-8))).astype(np.float64)

    cam = _build_camera(crop_size)

    # 1. Fit current pose
    ok, rvec, tvec = cv2.solvePnP(
        pts3d, lm68_crop.astype(np.float64), cam, _DIST,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None

    # 2. Re-project with zero rotation (frontal) — keep same translation/scale
    rvec_frontal = np.zeros((3, 1), dtype=np.float64)
    frontal_2d, _ = cv2.projectPoints(pts3d, rvec_frontal, tvec, cam, _DIST)
    return frontal_2d.reshape(-1, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Frontalization helpers
# ---------------------------------------------------------------------------

def frontalize_crop(
    aligned_img: np.ndarray,       # (H, W, 3) BGR  — align_crop output
    actual_lm68: np.ndarray,       # (68, 2) float32  — landmarks in crop space
    frontal_lm68: Optional[np.ndarray] = None,  # override; None → auto-compute
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Warp *aligned_img* so that its 68 landmarks match the frontal reference.

    Parameters
    ----------
    aligned_img   : BGR face crop (H × W × 3, uint8 or float32).
    actual_lm68   : 68-point 2-D landmarks in the crop's coordinate space.
    frontal_lm68  : Pre-computed frontal positions (crop space).
                    If None, computed automatically via solvePnP + re-projection.

    Returns
    -------
    (frontalized_img, M_forward)
        ``M_forward`` is the 2×3 float64 affine that maps actual → frontal.
        Returns ``(aligned_img, None)`` if estimation fails.
    """
    h, w = aligned_img.shape[:2]

    if frontal_lm68 is None:
        frontal_lm68 = get_frontal_landmarks_from_pose(actual_lm68, h)

    if frontal_lm68 is None:
        return aligned_img, None

    try:
        M, inliers = cv2.estimateAffine2D(
            actual_lm68.astype(np.float32),
            frontal_lm68.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=8.0,
            maxIters=2000,
            confidence=0.99,
        )
    except Exception:
        return aligned_img, None

    if M is None:
        return aligned_img, None

    frontalized = cv2.warpAffine(
        aligned_img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return frontalized, M


def defrontalize_crop(
    swapped_img: np.ndarray,        # (H, W, 3) BGR — swap output in frontal space
    M_forward: np.ndarray,          # 2×3 affine (actual → frontal)
) -> np.ndarray:
    """
    Apply the inverse of *M_forward* to restore the original pose.

    ``M_forward`` was returned by :func:`frontalize_crop`.  The inverse maps
    frontal → actual, putting the swapped face back in the original crop pose.
    """
    h, w = swapped_img.shape[:2]
    M_inv = cv2.invertAffineTransform(M_forward)
    return cv2.warpAffine(
        swapped_img, M_inv, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


# ---------------------------------------------------------------------------
# Pose-threshold helper (re-exported for convenience)
# ---------------------------------------------------------------------------

def should_frontalize(
    lm68_crop: np.ndarray,
    crop_size: int,
    yaw_threshold_deg: float = 25.0,
    pitch_threshold_deg: float = 25.0,
) -> Tuple[bool, float, float]:
    """
    Return (should, yaw_deg, pitch_deg).

    Uses ``estimate_pose`` / ``decompose_yaw_pitch`` from face_3d_recon
    to determine whether the crop is angled enough to warrant frontalization.
    """
    from roop.face_3d_recon import estimate_pose, decompose_yaw_pitch
    try:
        rvec, _ = estimate_pose(lm68_crop, crop_size)
        yaw, pitch = decompose_yaw_pitch(rvec)
        yd, pd = math.degrees(yaw), math.degrees(pitch)
        return (abs(yd) > yaw_threshold_deg or abs(pd) > pitch_threshold_deg), yd, pd
    except Exception:
        return False, 0.0, 0.0
