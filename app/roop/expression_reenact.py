"""
expression_reenact.py — Landmark-based expression transfer (TPS warp)

Uses the insightface 106-pt 2-D landmarks that buffalo_l already provides to
warp the post-swap face crop toward one of the built-in expression presets.
No new model downloads are required — only scipy (already installed) and
OpenCV are used.

Workflow
--------
1. Once, when using a named preset:
   • ``prepare_preset(name)`` returns a dict with ``type='preset'`` and a
     pre-defined ``delta_512`` array (106, 2) float32 in canonical 512-px
     space.  No face detection is needed.

2. Per swapped-face crop (at ``subsample_size × subsample_size``):
   • Map the TARGET face's 106 landmarks into crop space via the affine M.
   • Compute per-landmark displacements (from image delta or preset delta).
   • Zero-out anchor landmarks (the convex-hull / face-boundary points) so
     the face outline does not drift — only interior expression features
     (mouth, eyebrows, eye openings) are warped.
   • Scale deltas by ``strength``.
   • Evaluate a Thin Plate Spline (TPS) interpolation of the displacement
     field on a coarse 64×64 grid, upsample to full crop size, and remap.

Insightface buffalo_l 106-pt landmark index reference (canonical groups)
------------------------------------------------------------------------
Jaw / face outline : 0-32
Right eyebrow      : 43, 48, 49, 50, 51
Left eyebrow       : 101, 102, 103, 104, 105
Nose               : 72-86
Right eye          : 35-42
Left eye           : 89-96
Outer lip corners  : 52 (left/viewer-right), 61 (right/viewer-left)
Inner lip ring     : 53-70  (upper: 53,54,55,56,57; lower: 58,59,60; corners: 61,52; etc.)
Chin tip           : 16
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Built-in expression preset deltas  (canonical 512-px landmark space)
# ---------------------------------------------------------------------------
# Each entry is a sparse dict  {landmark_index: (dx, dy)}  where +y is DOWN.
# Only the landmarks that should actually move are listed; all others are 0.
# The convex-hull anchors are zeroed in _apply_tps regardless, so it is safe
# to list face-boundary indices here — they simply have no effect.
# ---------------------------------------------------------------------------

def _make_delta(sparse: dict) -> np.ndarray:
    """Convert a sparse {idx: (dx,dy)} dict to a (106,2) float32 delta array."""
    d = np.zeros((106, 2), dtype=np.float32)
    for idx, (dx, dy) in sparse.items():
        d[idx] = (dx, dy)
    return d


# landmark groups used across presets (insightface 106-pt, 512-px space)
_MOUTH_CORNER_L = 52   # viewer's right, subject's left
_MOUTH_CORNER_R = 61   # viewer's left, subject's right
_INNER_LIP_LOWER = [58, 59, 60]   # bottom of inner lip ring
_INNER_LIP_UPPER = [53, 54, 55, 56, 57]  # top of inner lip ring
_LOWER_LIP_MID   = 66   # mid lower lip
_UPPER_LIP_MID   = 62   # mid upper lip
_CHIN            = 16   # chin tip

_R_BROW_INNER = 50    # right brow, inner end  (near nose bridge)
_L_BROW_INNER = 101   # left brow,  inner end
_R_BROW_OUTER = 43    # right brow, outer end
_L_BROW_OUTER = 102   # left brow,  outer end
_R_BROW_MID   = 48    # right brow, middle
_L_BROW_MID   = 103   # left brow,  middle


EXPRESSION_PRESETS: dict[str, np.ndarray] = {

    # ── Happy ────────────────────────────────────────────────────────────
    # Corners up, lower lip pulled up slightly, cheek push negligible.
    'Happy': _make_delta({
        _MOUTH_CORNER_L:  (-8, -20),
        _MOUTH_CORNER_R:  ( 8, -20),
        58:  ( 0,  8),   # lower inner lip mid-left
        59:  ( 0, 10),   # lower inner lip center
        60:  ( 0,  8),   # lower inner lip mid-right
        _LOWER_LIP_MID:   ( 0,  8),
        # brows rise slightly with a smile
        _R_BROW_MID:      ( 0, -6),
        _L_BROW_MID:      ( 0, -6),
    }),

    # ── Sad ─────────────────────────────────────────────────────────────
    # Corners down, inner brows raised (worried crease), outer brows lower.
    'Sad': _make_delta({
        _MOUTH_CORNER_L:  ( 5, 18),
        _MOUTH_CORNER_R:  (-5, 18),
        _LOWER_LIP_MID:   ( 0, -6),   # pout
        _R_BROW_INNER:    ( 8,-16),    # inner brow up + toward center
        _L_BROW_INNER:    (-8,-16),
        _R_BROW_OUTER:    ( 0,  6),    # outer brow droop
        _L_BROW_OUTER:    ( 0,  6),
    }),

    # ── Angry ────────────────────────────────────────────────────────────
    # Inner brows strongly down+together, outer brows level, corners down.
    'Angry': _make_delta({
        _R_BROW_INNER:    (14, 16),    # push inward + down
        _L_BROW_INNER:    (-14, 16),
        _R_BROW_OUTER:    ( 0, -4),
        _L_BROW_OUTER:    ( 0, -4),
        _MOUTH_CORNER_L:  ( 4, 10),
        _MOUTH_CORNER_R:  (-4, 10),
    }),

    # ── Surprised ────────────────────────────────────────────────────────
    # Brows shoot up, jaw drops, inner lips open.
    'Surprised': _make_delta({
        _R_BROW_INNER:    ( 0,-20),
        _L_BROW_INNER:    ( 0,-20),
        _R_BROW_MID:      ( 0,-20),
        _L_BROW_MID:      ( 0,-20),
        _R_BROW_OUTER:    ( 0,-16),
        _L_BROW_OUTER:    ( 0,-16),
        _CHIN:            ( 0, 28),    # jaw drops
        59:               ( 0, 22),   # lower inner lip follows jaw
        _LOWER_LIP_MID:   ( 0, 20),
        _UPPER_LIP_MID:   ( 0,-10),   # upper lip retracts up slightly
    }),

    # ── Fear ─────────────────────────────────────────────────────────────
    # Like Surprised but inner brows pulled together + up; mouth wide.
    'Fear': _make_delta({
        _R_BROW_INNER:    (10,-18),
        _L_BROW_INNER:    (-10,-18),
        _R_BROW_MID:      ( 2,-14),
        _L_BROW_MID:      (-2,-14),
        _R_BROW_OUTER:    ( 0, -8),
        _L_BROW_OUTER:    ( 0, -8),
        _MOUTH_CORNER_L:  (-6, 12),
        _MOUTH_CORNER_R:  ( 6, 12),
        _CHIN:            ( 0, 18),
        59:               ( 0, 14),
    }),

    # ── Disgusted ───────────────────────────────────────────────────────
    # Upper lip curled up (unilateral), nose wrinkle area, brows slightly down.
    'Disgusted': _make_delta({
        _UPPER_LIP_MID:   ( 0,-14),   # upper lip rises in center
        53:               ( 0,-10),
        57:               ( 0,-10),
        _MOUTH_CORNER_L:  (-4,  6),
        _MOUTH_CORNER_R:  ( 4,  6),
        _R_BROW_INNER:    ( 4,  8),
        _L_BROW_INNER:    (-4,  8),
        _R_BROW_MID:      ( 0,  6),
        _L_BROW_MID:      ( 0,  6),
    }),
}


class ExpressionReenactor:
    """Singleton for landmark-based expression warping.

    Use ``ExpressionReenactor.instance()`` rather than constructing directly.
    """

    _singleton: Optional['ExpressionReenactor'] = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> 'ExpressionReenactor':
        if cls._singleton is None:
            with cls._lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_preset(self, name: str) -> Optional[dict]:
        """
        Return a driving_data dict for a named built-in expression preset.

        Parameters
        ----------
        name
            One of the keys in ``EXPRESSION_PRESETS`` (e.g. ``'Happy'``).

        Returns
        -------
        dict with ``'type': 'preset'`` and ``'delta_512': (106,2) float32``,
        or ``None`` if the name is not recognised.
        """
        delta = EXPRESSION_PRESETS.get(name)
        if delta is None:
            print(f"[ExpressionReenactor] Unknown preset '{name}'. "
                  f"Available: {list(EXPRESSION_PRESETS)}")
            return None
        return {
            'type':      'preset',
            'delta_512': delta.copy(),   # (106, 2) float32
        }

    def apply(
        self,
        crop: np.ndarray,                 # (S, S, 3) uint8 BGR swapped face crop
        source_lm106_crop: np.ndarray,    # (106, 2) float32 — target-face lm in crop space
        driving_data: dict,               # from prepare_preset()
        strength: float,                  # 0.0 – 1.0
        crop_size: int,                   # S  (== crop.shape[0])
    ) -> np.ndarray:
        """
        Warp *crop* so that its expression matches the driving reference.

        Returns the warped crop (same shape and dtype).
        Falls back to the unchanged *crop* on any error.
        """
        if driving_data is None or strength <= 0.0:
            return crop
        try:
            return self._apply_tps(crop, source_lm106_crop, driving_data,
                                   strength, crop_size)
        except Exception as e:
            print(f"[ExpressionReenactor] apply failed: {e}")
            return crop

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _apply_tps(
        self,
        crop: np.ndarray,
        source_lm_crop: np.ndarray,     # (106, 2) in crop-pixel coords
        driving_data: dict,
        strength: float,
        crop_size: int,
    ) -> np.ndarray:
        from scipy.interpolate import RBFInterpolator

        scale = crop_size / 512.0

        # Preset path: delta is pre-defined in 512-px space; scale it.
        delta_512: np.ndarray = driving_data['delta_512']   # (106, 2)
        delta = (delta_512 * scale).astype(np.float64)

        # Compute hull from SOURCE landmarks (no driving image available).
        hull_idx: list = (
            cv2.convexHull(source_lm_crop.astype(np.int32), returnPoints=False)
            .flatten()
            .tolist()
        )

        # Zero out anchor (hull) points so the face boundary stays fixed
        delta[hull_idx] = 0.0

        # Apply user strength
        delta *= strength

        # Control points in (x, y) order — matches landmark coordinate convention
        ctrl_pts = source_lm_crop.astype(np.float64)   # (106, 2)

        # Fit independent TPS RBFs for Δx and Δy
        rbf_x = RBFInterpolator(ctrl_pts, delta[:, 0],
                                 kernel='thin_plate_spline', smoothing=0.0)
        rbf_y = RBFInterpolator(ctrl_pts, delta[:, 1],
                                 kernel='thin_plate_spline', smoothing=0.0)

        # Evaluate on a coarse 64×64 grid for speed, then upsample to crop_size.
        # Landmark positions are (x=col, y=row) so we build the grid accordingly.
        COARSE = 64
        gx  = np.linspace(0.0, crop_size - 1.0, COARSE)   # column  (x) coords
        gy  = np.linspace(0.0, crop_size - 1.0, COARSE)   # row     (y) coords
        gxx, gyy = np.meshgrid(gx, gy)                    # gxx[r,c]=gx[c], gyy[r,c]=gy[r]
        # grid_pts rows are (x, y) — matching ctrl_pts convention
        grid_pts = np.column_stack([gxx.ravel(), gyy.ravel()])   # (COARSE², 2)

        dx_coarse = rbf_x(grid_pts).reshape(COARSE, COARSE).astype(np.float32)
        dy_coarse = rbf_y(grid_pts).reshape(COARSE, COARSE).astype(np.float32)

        # Upsample displacement field to full crop_size
        dx = cv2.resize(dx_coarse, (crop_size, crop_size),
                        interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(dy_coarse, (crop_size, crop_size),
                        interpolation=cv2.INTER_LINEAR)

        # Build inverse remap arrays.
        # cv2.remap(src, map_x, map_y): output[r,c] = src[map_y[r,c], map_x[r,c]]
        # We have FORWARD flow (source → target), so for the inverse:
        #   map_x[r,c] = c − dx[r,c]   (x-col to sample from source)
        #   map_y[r,c] = r − dy[r,c]   (y-row to sample from source)
        # This is the standard small-displacement approximation.
        ox = np.arange(crop_size, dtype=np.float32)
        oy = np.arange(crop_size, dtype=np.float32)
        map_x_grid, map_y_grid = np.meshgrid(ox, oy)   # (crop_size, crop_size) each
        map_x = (map_x_grid - dx).astype(np.float32)
        map_y = (map_y_grid - dy).astype(np.float32)

        warped = cv2.remap(crop, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
        return warped
