import glob
import os

import cv2
import gradio as gr
import numpy as np
from PIL import Image

import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import roop.globals

RESOLUTION_CHOICES = ["1280x720", "1920x1080", "854x480", "3840x2160"]
ROTATION_CHOICES   = [
    "None (no change)",
    "90° Clockwise", "90° Counter-clockwise",
    "180°",
    "Flip Horizontal", "Flip Vertical",
]
ROTATE_FILTERS = {
    "90° Clockwise":        ["transpose=1"],
    "90° Counter-clockwise": ["transpose=2"],
    "180°":                  ["vflip", "hflip"],
    "Flip Horizontal":       ["hflip"],
    "Flip Vertical":         ["vflip"],
}


def extras_tab(bt_destfiles=None):
    # State: tracks detected properties of the current file
    file_info = gr.State({"width": 0, "height": 0, "fps": 24.0, "is_video": False})

    with gr.Tab("✏️ Editor"):

        # ── Upload + Preview ──────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                files_to_process = gr.Files(
                    label="Upload file",
                    file_count="multiple",
                    file_types=["image", "video", ".webp"],
                )
            with gr.Column(scale=2):
                preview_image = gr.Image(
                    label="Preview", visible=False, interactive=False,
                    show_download_button=False,
                )
                preview_video = gr.Video(
                    label="Preview", visible=False, interactive=False,
                )

        # ── Operations ────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Group():
                gr.Markdown("#### Resolution")
                current_res_label = gr.Markdown("**Current:** —")
                resize_resolution = gr.Dropdown(
                    RESOLUTION_CHOICES, value=RESOLUTION_CHOICES[0],
                    label="Target", show_label=False,
                )

            with gr.Group():
                gr.Markdown("#### Rotate / Flip")
                rotation_choice = gr.Dropdown(
                    ROTATION_CHOICES, value="None (no change)",
                    label="Transform", show_label=False,
                )

            with gr.Group(visible=False) as fps_group:
                gr.Markdown("#### Change FPS")
                current_fps_label = gr.Markdown("**Current:** —")
                fps_value = gr.Slider(1, 120, value=30, step=1,
                                      label="Target FPS", show_label=False)

        # ── Crop ──────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("#### Crop  *(trim from each edge as % of frame size)*")
            with gr.Row():
                crop_left   = gr.Slider(0, 49, value=0, step=1, label="Left %")
                crop_right  = gr.Slider(0, 49, value=0, step=1, label="Right %")
                crop_top    = gr.Slider(0, 49, value=0, step=1, label="Top %")
                crop_bottom = gr.Slider(0, 49, value=0, step=1, label="Bottom %")

        # ── Single Apply ──────────────────────────────────────────────
        with gr.Row():
            btn_apply = gr.Button("Apply", variant="primary")

        # ── Output preview ────────────────────────────────────────────
        with gr.Row():
            output_image = gr.Image(
                label="Output", visible=False, interactive=False,
                show_download_button=True,
            )
            output_video = gr.Video(
                label="Output", visible=False, interactive=False,
            )

        with gr.Row():
            send_to_faceswap_btn = gr.Button(
                "↗ Send to Face Swap", size="sm",
                visible=bt_destfiles is not None,
            )

    # Holds the output path(s) for Send to Face Swap
    output_path_state = gr.State(None)

    # ══════════════════════════════════════════════════════════════════════
    # 🎞️ Frame Editor tab  (hidden — preserved for future use)
    # ══════════════════════════════════════════════════════════════════════
    with gr.Tab("🎞️ Frame Editor", visible=False):
        # Persistent state for the loaded frame set
        fe_frames_list = gr.State([])   # sorted processed frame paths (_frames/)
        fe_orig_list   = gr.State([])   # sorted original (unswapped) frame paths (_frames_orig/)
        fe_orig_dir    = gr.State("")   # absolute path to _frames_orig/ directory
        fe_meta        = gr.State({})   # metadata dict (fps, source, image_format)

        # ── Directory loader ──────────────────────────────────────────
        with gr.Row():
            fe_dir_input = gr.Textbox(
                label="Frames directory",
                placeholder="Paste the path to a _frames folder, e.g. C:/output/myvideo_frames",
                scale=5,
            )
            fe_load_btn = gr.Button("📂 Load", variant="primary", scale=1, min_width=80)

        fe_status = gr.Markdown("_No frames loaded — paste a frames directory path and click Load._")

        # ── Frame navigation ─────────────────────────────────────────
        with gr.Row():
            fe_prev_btn = gr.Button("◀ Prev", size="sm", scale=1, min_width=80)
            fe_slider   = gr.Slider(minimum=1, maximum=1, value=1, step=1,
                                    label="Frame", scale=8)
            fe_next_btn = gr.Button("Next ▶", size="sm", scale=1, min_width=80)

        # ── Frame view + Mask controls ───────────────────────────────
        with gr.Row():
            # Left column: frame view
            with gr.Column(scale=3):
                fe_frame_view = gr.Image(
                    label="Current frame (processed)",
                    interactive=False,
                    height=520,
                    value=None,
                )

            # Right column: per-frame mask settings
            with gr.Column(scale=2):
                gr.Markdown("#### Per-Frame Mask Settings")

                fe_mask_top = gr.Slider(
                    0, 2.0, value=roop.globals.CFG.mask_top,
                    label="Offset Face Top", step=0.01, interactive=True,
                )
                fe_mask_bottom = gr.Slider(
                    0, 2.0, value=roop.globals.CFG.mask_bottom,
                    label="Offset Face Bottom", step=0.01, interactive=True,
                )
                fe_mask_left = gr.Slider(
                    0, 2.0, value=roop.globals.CFG.mask_left,
                    label="Offset Face Left", step=0.01, interactive=True,
                )
                fe_mask_right = gr.Slider(
                    0, 2.0, value=roop.globals.CFG.mask_right,
                    label="Offset Face Right", step=0.01, interactive=True,
                )
                fe_face_blend = gr.Slider(
                    0, 200, value=roop.globals.CFG.face_mask_blend,
                    label="Face Mask Edge Blend", step=1, interactive=True,
                )
                fe_mouth_blend = gr.Slider(
                    0, 200, value=roop.globals.CFG.mouth_mask_blend,
                    label="Mouth Mask Blend", step=1, interactive=True,
                )
                with gr.Row():
                    fe_mouth_top = gr.Slider(
                        0, 10.0, value=roop.globals.CFG.mouth_top_scale,
                        label="Mouth Top", step=0.1, interactive=True,
                    )
                    fe_mouth_bottom = gr.Slider(
                        0, 10.0, value=roop.globals.CFG.mouth_bottom_scale,
                        label="Mouth Bottom", step=0.1, interactive=True,
                    )
                with gr.Row():
                    fe_mouth_left = gr.Slider(
                        0, 10.0, value=roop.globals.CFG.mouth_left_scale,
                        label="Mouth Left", step=0.1, interactive=True,
                    )
                    fe_mouth_right = gr.Slider(
                        0, 10.0, value=roop.globals.CFG.mouth_right_scale,
                        label="Mouth Right", step=0.1, interactive=True,
                    )

                fe_mask_btn = gr.Button(
                    "🎭 Edit Canvas Mask",
                    variant="secondary",
                )

                gr.Markdown(
                    "_Opens the mask editor using the original unswapped face crop. "
                    "Paint include/exclude areas, then click Apply & Close._"
                )

                fe_save_mask_btn = gr.Button("💾 Save Mask for this Frame", variant="primary")
                fe_mask_save_status = gr.Markdown("")

        # Hidden Gradio stores used as JS ↔ Python bridge for the canvas mask editor
        # These mirror the faceswap tab's stores but are scoped to the Frame Editor.
        fe_mask_json_store           = gr.Textbox(value="", visible=False,
                                                   elem_id="fe_mask_json_store",
                                                   label="fe_mask_json_store")
        fe_mask_face_crop_store      = gr.Textbox(value="", visible=False,
                                                   elem_id="fe_mask_face_crop_store",
                                                   label="fe_mask_face_crop_store")
        fe_mask_face_swap_crop_store = gr.Textbox(value="", visible=False,
                                                   elem_id="fe_mask_face_swap_crop_store",
                                                   label="fe_mask_face_swap_crop_store")

        # ── Compile (reprocess originals) ──────────────────────────────
        with gr.Row(variant="panel"):
            fe_fps = gr.Number(value=24.0, label="Output FPS",
                               minimum=1, maximum=120, scale=1, min_width=120)
            fe_compile_mp4_btn = gr.Button("🎬 Reprocess → MP4", scale=2)
            fe_compile_gif_btn = gr.Button("🎞️ Reprocess → GIF", scale=2)

        gr.Markdown(
            "_Reprocess iterates each **original** (unswapped) frame using the saved per-frame "
            "mask settings and the current face-swap configuration, then compiles the result._"
        )

        # ── Compiled output preview ───────────────────────────────────
        with gr.Row():
            fe_out_image = gr.Image(label="Reprocessed output",
                                    visible=False, interactive=False,
                                    show_download_button=True)
            fe_out_video = gr.Video(label="Reprocessed output",
                                    visible=False, interactive=False)

    # ── All slider components for mask I/O ───────────────────────────
    _fe_mask_sliders = [
        fe_mask_top, fe_mask_bottom, fe_mask_left, fe_mask_right,
        fe_face_blend,
        fe_mouth_blend, fe_mouth_top, fe_mouth_bottom,
        fe_mouth_left, fe_mouth_right,
    ]

    # ── Frame Editor event wiring ─────────────────────────────────────

    # Load button
    fe_load_btn.click(
        fn=on_fe_load,
        inputs=[fe_dir_input],
        outputs=[fe_slider, fe_status, fe_frames_list, fe_orig_list,
                 fe_orig_dir, fe_meta, fe_fps],
        show_progress="hidden",
    ).then(
        fn=on_fe_frame_changed,
        inputs=[fe_slider, fe_frames_list, fe_orig_list, fe_orig_dir],
        outputs=[fe_frame_view, fe_mask_json_store,
                 fe_mask_face_crop_store, fe_mask_face_swap_crop_store,
                 *_fe_mask_sliders],
        show_progress="hidden",
    )

    # Slider release
    fe_slider.release(
        fn=on_fe_frame_changed,
        inputs=[fe_slider, fe_frames_list, fe_orig_list, fe_orig_dir],
        outputs=[fe_frame_view, fe_mask_json_store,
                 fe_mask_face_crop_store, fe_mask_face_swap_crop_store,
                 *_fe_mask_sliders],
        show_progress="hidden",
    )

    # Prev / Next buttons
    fe_prev_btn.click(
        fn=on_fe_prev_frame,
        inputs=[fe_slider, fe_frames_list, fe_orig_list, fe_orig_dir],
        outputs=[fe_slider, fe_frame_view, fe_mask_json_store,
                 fe_mask_face_crop_store, fe_mask_face_swap_crop_store,
                 *_fe_mask_sliders],
        show_progress="hidden",
    )
    fe_next_btn.click(
        fn=on_fe_next_frame,
        inputs=[fe_slider, fe_frames_list, fe_orig_list, fe_orig_dir],
        outputs=[fe_slider, fe_frame_view, fe_mask_json_store,
                 fe_mask_face_crop_store, fe_mask_face_swap_crop_store,
                 *_fe_mask_sliders],
        show_progress="hidden",
    )

    # Canvas mask button — triggers JS maskToggleFrameEditor()
    fe_mask_btn.click(
        fn=None,
        js="() => maskToggleFrameEditor()",
    )

    # Save mask for current frame
    fe_save_mask_btn.click(
        fn=on_fe_save_mask,
        inputs=[fe_slider, fe_frames_list, fe_orig_dir,
                *_fe_mask_sliders,
                fe_mask_json_store],
        outputs=[fe_mask_save_status],
    )

    # Compile / reprocess buttons
    fe_compile_mp4_btn.click(
        fn=on_fe_compile_mp4,
        inputs=[fe_frames_list, fe_orig_list, fe_orig_dir, fe_meta, fe_fps],
        outputs=[fe_out_image, fe_out_video, fe_status],
    )
    fe_compile_gif_btn.click(
        fn=on_fe_compile_gif,
        inputs=[fe_frames_list, fe_orig_list, fe_orig_dir, fe_meta, fe_fps],
        outputs=[fe_out_image, fe_out_video, fe_status],
    )

    # ── Event wiring ──────────────────────────────────────────────────
    files_to_process.clear(
        fn=on_file_clear,
        outputs=[
            preview_image, preview_video,
            output_image, output_video,
            output_path_state,
        ],
        show_progress="hidden",
    )

    files_to_process.upload(
        fn=on_file_upload,
        inputs=[files_to_process],
        outputs=[
            preview_image, preview_video,
            current_res_label, resize_resolution,
            current_fps_label, fps_value,
            fps_group,
            file_info,
        ],
        show_progress="hidden",
    )

    btn_apply.click(
        fn=on_apply_all,
        inputs=[
            files_to_process,
            resize_resolution, rotation_choice,
            fps_value,
            crop_left, crop_right, crop_top, crop_bottom,
            file_info,
        ],
        outputs=[output_image, output_video, output_path_state],
    )

    if bt_destfiles is not None:
        send_to_faceswap_btn.click(
            fn=on_send_to_faceswap,
            inputs=[output_path_state],
            outputs=[bt_destfiles],
        )


# ── Handlers ──────────────────────────────────────────────────────────

def on_file_clear():
    hidden = gr.update(visible=False, value=None)
    return hidden, hidden, hidden, hidden, None


def on_file_upload(files):
    empty = (
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
        gr.update(value="**Current:** —"),
        gr.update(choices=RESOLUTION_CHOICES, value=RESOLUTION_CHOICES[0]),
        gr.update(value="**Current:** —"),
        gr.update(value=30),
        gr.update(visible=False),
        {"width": 0, "height": 0, "fps": 24.0, "is_video": False},
    )
    if not files:
        return empty

    path = files[0].name if hasattr(files[0], 'name') else str(files[0])
    is_awebp   = util.is_animated_webp(path)
    is_agif    = util.is_animated_gif(path)
    is_animated = is_awebp or is_agif
    is_img = util.is_image(path)   # returns False for animated webp/gif
    is_vid = util.is_video(path) or is_animated

    if not is_img and not is_vid:
        return empty

    # Detect properties
    w, h = util.detect_dimensions(path)
    if is_vid and not is_animated:
        fps = util.detect_fps(path)
    elif is_animated:
        fps = util.detect_fps(path)  # PIL-based for webp; cv2-based for gif
    else:
        fps = 24.0

    # Build resolution dropdown choices with current res at top
    current_res = f"{w}x{h}" if w and h else RESOLUTION_CHOICES[0]
    choices = [current_res] + [r for r in RESOLUTION_CHOICES if r != current_res]

    info = {"width": w, "height": h, "fps": fps, "is_video": is_vid, "is_animated_gif": is_agif, "is_animated_webp": is_awebp}

    # Animated webp/gif previews in the image component (browsers render them natively)
    show_as_img = is_img or is_animated
    show_as_vid = is_vid and not is_animated
    return (
        gr.update(visible=show_as_img, value=path if show_as_img else None),
        gr.update(visible=show_as_vid, value=path if show_as_vid else None),
        gr.update(value=f"**Current:** {w} × {h}"),
        gr.update(choices=choices, value=current_res),
        gr.update(value=f"**Current:** {fps:.2f} fps"),
        gr.update(value=round(fps)),
        gr.update(visible=is_vid),
        info,
    )


def on_apply_all(files, resolution, rotation, fps,
                 crop_left, crop_right, crop_top, crop_bottom,
                 file_info):
    no_output = (
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
        None,
    )
    print(f"[Editor] on_apply_all called: files={len(files) if files else 0}, rotation={rotation!r}, file_info={file_info}")
    if not files:
        print("[Editor] No files — aborting")
        return no_output

    paths = [f.name if hasattr(f, 'name') else str(f) for f in files]
    is_vid      = file_info.get("is_video", False)
    is_agif     = file_info.get("is_animated_gif", False)
    is_awebp    = file_info.get("is_animated_webp", False)
    cur_w  = file_info.get("width", 0)
    cur_h  = file_info.get("height", 0)
    cur_fps = file_info.get("fps", 24.0)
    print(f"[Editor] paths={paths}, is_vid={is_vid}, is_agif={is_agif}, cur_w={cur_w}, cur_h={cur_h}, cur_fps={cur_fps}")

    # Build vf filter list (order: crop → rotate → scale → fps)
    # Note: for animated GIF, fps filter is embedded inside apply_media_transforms_gif
    filters = []

    if any(v > 0 for v in [crop_left, crop_right, crop_top, crop_bottom]):
        l, r, t, b = crop_left/100, crop_right/100, crop_top/100, crop_bottom/100
        filters.append(
            f"crop=in_w*(1-{l:.4f}-{r:.4f}):in_h*(1-{t:.4f}-{b:.4f})"
            f":in_w*{l:.4f}:in_h*{t:.4f}"
        )

    if rotation in ROTATE_FILTERS:
        filters.extend(ROTATE_FILTERS[rotation])

    target_w, target_h = (int(x) for x in resolution.split('x'))
    if target_w != cur_w or target_h != cur_h:
        filters.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
        )

    # FPS filter — not needed for animated GIF (handled by apply_media_transforms_gif)
    if is_vid and not is_agif and abs(fps - cur_fps) > 0.1:
        filters.append(f"fps={fps}")

    print(f"[Editor] filters built: {filters}")
    if not filters and not (is_agif and abs(fps - cur_fps) > 0.1):
        gr.Info("No changes to apply.")
        return no_output

    out = []
    for f in paths:
        dest = util.get_destfilename_from_path(f, roop.globals.output_path, '_edited')
        if is_awebp or util.is_animated_webp(f):
            # Animated webp: FFmpeg can't decode it — pipe PIL frames through ffmpeg.
            # Output as mp4 since libx264 cannot write to a .webp container.
            dest = os.path.splitext(dest)[0] + '.mp4'
            success = ffmpeg.apply_media_transforms_webp(f, dest, filters, cur_fps)
        elif is_agif or util.is_animated_gif(f):
            # Animated GIF: use palettegen+paletteuse pipeline to preserve quality.
            target_fps = fps if abs(fps - cur_fps) > 0.1 else None
            success = ffmpeg.apply_media_transforms_gif(f, dest, filters, target_fps)
        else:
            success = ffmpeg.apply_media_transforms(f, dest, filters, is_vid)
        if success:
            out.append(dest)
        else:
            gr.Warning(f'Processing failed for {os.path.basename(f)}')

    if not out:
        return no_output

    first = out[0]
    # Show in image component if static image OR animated gif/webp
    # (browsers play these natively in <img>; gr.Video doesn't handle them well).
    if util.is_image(first) or util.is_animated_gif(first) or util.is_animated_webp(first):
        return gr.update(visible=True, value=first), gr.update(visible=False, value=None), out
    return gr.update(visible=False, value=None), gr.update(visible=True, value=first), out


def on_send_to_faceswap(paths):
    if not paths:
        return None
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Frame Editor handlers
# ══════════════════════════════════════════════════════════════════════════════

# Number of mask sliders returned by frame-change handlers
_FE_NUM_SLIDERS = 10

def _fe_scan_frames(frames_dir: str, image_format: str):
    """Return sorted list of frame image paths in *frames_dir* for *image_format*."""
    return sorted(glob.glob(os.path.join(frames_dir, f'*.{image_format}')))


def _fe_resolve_dirs(frames_dir: str):
    """Given a frames directory (either _frames or _frames_orig), return
    (proc_dir, orig_dir) where proc_dir is the processed frames directory
    and orig_dir is the unswapped originals directory.  Either may be absent."""
    frames_dir = frames_dir.rstrip('/\\')
    if frames_dir.endswith('_frames_orig'):
        orig_dir = frames_dir
        proc_dir = frames_dir[:-len('_orig')]  # strip '_orig' suffix → _frames
    else:
        proc_dir = frames_dir
        orig_dir = frames_dir + '_orig'        # append '_orig' → _frames_orig
    return proc_dir, orig_dir


def _fe_scan_dir(d: str, meta: dict):
    """Scan directory *d* for frame images. Returns (paths, image_format)."""
    if not d or not os.path.isdir(d):
        return [], meta.get('image_format', roop.globals.CFG.output_image_format)
    image_format = meta.get('image_format', roop.globals.CFG.output_image_format)
    paths = _fe_scan_frames(d, image_format)
    if not paths:
        for fmt in ('png', 'jpg', 'jpeg'):
            paths = _fe_scan_frames(d, fmt)
            if paths:
                image_format = fmt
                break
    return paths, image_format


def _fe_load_path(frame_num: int, frame_paths: list):
    """Return the file path for frame *frame_num* (1-indexed), or None."""
    if not frame_paths:
        return None
    idx = max(0, min(len(frame_paths) - 1, frame_num - 1))
    path = frame_paths[idx]
    return path if os.path.isfile(path) else None


def _fe_default_sliders():
    """Return default slider values (10 values) from global CFG."""
    cfg = roop.globals.CFG
    return [
        cfg.mask_top, cfg.mask_bottom, cfg.mask_left, cfg.mask_right,
        cfg.face_mask_blend,
        cfg.mouth_mask_blend, cfg.mouth_top_scale, cfg.mouth_bottom_scale,
        cfg.mouth_left_scale, cfg.mouth_right_scale,
    ]


def _fe_build_frame_outputs(frame_num: int, frame_paths: list,
                             orig_paths: list, orig_dir: str):
    """Build the full output tuple for a frame navigation event.

    Returns: (frame_view, mask_json, face_crop, swap_crop, *10_slider_values)
    """
    # Processed frame for display
    proc_path = _fe_load_path(frame_num, frame_paths)
    if proc_path is None and orig_paths:
        proc_path = _fe_load_path(frame_num, orig_paths)

    # Generate face crops in a background-safe way
    face_crop_url = ""
    swap_crop_url = ""
    try:
        from roop.core import get_face_crop_from_frame
        # Original frame → mask editor background
        orig_path = _fe_load_path(frame_num, orig_paths) if orig_paths else None
        if orig_path and os.path.isfile(orig_path):
            orig_bgr = cv2.imread(orig_path)
            if orig_bgr is not None:
                face_crop_url = get_face_crop_from_frame(orig_bgr)
        # Processed frame → live preview background in mask editor
        if proc_path and os.path.isfile(proc_path):
            proc_bgr = cv2.imread(proc_path)
            if proc_bgr is not None:
                swap_crop_url = get_face_crop_from_frame(proc_bgr)
    except Exception as e:
        print(f"[FrameEditor] face crop error: {e}")

    # Load per-frame mask sidecar
    mask_json = ""
    slider_vals = _fe_default_sliders()
    if orig_dir and orig_paths:
        basename = os.path.basename(_fe_load_path(frame_num, orig_paths) or "")
        if basename:
            mask_data = util.load_frame_mask(orig_dir, basename)
            if mask_data:
                mask_json = mask_data.get('mask_json', '')
                slider_vals = [
                    mask_data.get('top',         slider_vals[0]),
                    mask_data.get('bottom',      slider_vals[1]),
                    mask_data.get('left',        slider_vals[2]),
                    mask_data.get('right',       slider_vals[3]),
                    mask_data.get('face_mask_blend',  slider_vals[4]),
                    mask_data.get('mouth_mask_blend', slider_vals[5]),
                    mask_data.get('mouth_top',        slider_vals[6]),
                    mask_data.get('mouth_bottom',     slider_vals[7]),
                    mask_data.get('mouth_left',       slider_vals[8]),
                    mask_data.get('mouth_right',      slider_vals[9]),
                ]

    return (
        gr.update(value=proc_path),
        gr.update(value=mask_json),
        gr.update(value=face_crop_url),
        gr.update(value=swap_crop_url),
        *[gr.update(value=v) for v in slider_vals],
    )


def on_fe_load(frames_dir: str):
    """Scan *frames_dir* (and its _orig counterpart), populate state, return updates."""
    _empty = (
        gr.update(value=1, minimum=1, maximum=1),
        "_No frames loaded — paste a frames directory path and click Load._",
        [], [], "", {},
        gr.update(value=24.0),
    )

    frames_dir = (frames_dir or '').strip().rstrip('/\\')
    if not frames_dir or not os.path.isdir(frames_dir):
        return _empty

    proc_dir, orig_dir = _fe_resolve_dirs(frames_dir)

    # Prefer the proc_dir for metadata; fall back to orig_dir
    meta_dir = proc_dir if os.path.isdir(proc_dir) else orig_dir
    meta     = util.read_frames_metadata(meta_dir) if os.path.isdir(meta_dir) else {}
    fps      = float(meta.get('fps', 24.0))

    proc_paths, image_format = _fe_scan_dir(proc_dir, meta)
    if image_format:
        meta['image_format'] = image_format
    orig_paths, _           = _fe_scan_dir(orig_dir, meta)

    all_paths = proc_paths or orig_paths
    if not all_paths:
        return (
            gr.update(value=1, minimum=1, maximum=1),
            "⚠️ No frame images found in this directory.",
            [], [], orig_dir if os.path.isdir(orig_dir) else "",
            meta,
            gr.update(value=fps),
        )

    n = len(all_paths)
    has_orig = bool(orig_paths)
    orig_note = " &nbsp;|&nbsp; ✅ originals found" if has_orig else " &nbsp;|&nbsp; ⚠️ no originals (_frames_orig not found)"
    status = (
        f"✅ **{n}** frames loaded &nbsp;|&nbsp; {fps:.2f} fps "
        f"&nbsp;|&nbsp; {meta.get('image_format', 'png')}{orig_note}"
    )
    return (
        gr.update(value=1, minimum=1, maximum=n),
        status,
        proc_paths,
        orig_paths,
        orig_dir if os.path.isdir(orig_dir) else "",
        meta,
        gr.update(value=fps),
    )


def on_fe_frame_changed(frame_num, frame_paths: list, orig_paths: list, orig_dir: str):
    """Load frame *frame_num*, generate face crops, load mask sidecar."""
    return _fe_build_frame_outputs(int(frame_num), frame_paths, orig_paths, orig_dir or "")


def on_fe_prev_frame(frame_num, frame_paths: list, orig_paths: list, orig_dir: str):
    """Navigate one frame backward."""
    n       = max(len(frame_paths), len(orig_paths), 1)
    new_num = max(1, int(frame_num) - 1)
    return (new_num, *_fe_build_frame_outputs(new_num, frame_paths, orig_paths, orig_dir or ""))


def on_fe_next_frame(frame_num, frame_paths: list, orig_paths: list, orig_dir: str):
    """Navigate one frame forward."""
    n       = max(len(frame_paths), len(orig_paths), 1)
    new_num = min(n, int(frame_num) + 1)
    return (new_num, *_fe_build_frame_outputs(new_num, frame_paths, orig_paths, orig_dir or ""))


def on_fe_save_mask(frame_num, frame_paths: list, orig_dir: str,
                    mask_top, mask_bottom, mask_left, mask_right,
                    face_blend, mouth_blend,
                    mouth_top, mouth_bottom, mouth_left, mouth_right,
                    mask_json: str):
    """Persist per-frame mask settings (sliders + canvas JSON) to sidecar file."""
    if not orig_dir or not os.path.isdir(orig_dir):
        return "⚠️ No originals directory found — run the face swap with 'Keep Frames' enabled first."

    # Find the corresponding orig frame basename
    n   = int(frame_num)
    idx = max(0, n - 1)
    # Use the frame number to reconstruct the expected filename (000001.png etc.)
    # Prefer matching from frame_paths, fall back to synthesising the name.
    basename = None
    all_files = sorted(glob.glob(os.path.join(orig_dir, '*.*')))
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if image_files and idx < len(image_files):
        basename = os.path.basename(image_files[idx])
    elif frame_paths and idx < len(frame_paths):
        basename = os.path.basename(frame_paths[idx])
    else:
        basename = f"{n:06d}.png"

    mask_data = {
        'top':              float(mask_top),
        'bottom':           float(mask_bottom),
        'left':             float(mask_left),
        'right':            float(mask_right),
        'face_mask_blend':  float(face_blend),
        'mouth_mask_blend': float(mouth_blend),
        'mouth_top':        float(mouth_top),
        'mouth_bottom':     float(mouth_bottom),
        'mouth_left':       float(mouth_left),
        'mouth_right':      float(mouth_right),
        'mask_json':        (mask_json or '').strip(),
    }
    util.save_frame_mask(orig_dir, basename, mask_data)
    return f"✅ Mask saved for frame {n} ({basename})"


def _fe_output_dir(frame_paths: list, orig_paths: list) -> str:
    """Return the output directory to write compiled files into."""
    out = roop.globals.output_path
    if not out:
        ref = frame_paths or orig_paths
        if ref:
            out = os.path.dirname(os.path.dirname(ref[0]))  # parent of frames dir
    return out or '.'


def _fe_apply_mask_to_facesets(mask_data: dict):
    """Temporarily set mask_offsets on all faces in INPUT_FACESETS from *mask_data*."""
    offsets = [
        mask_data.get('top',              roop.globals.CFG.mask_top),
        mask_data.get('bottom',           roop.globals.CFG.mask_bottom),
        mask_data.get('left',             roop.globals.CFG.mask_left),
        mask_data.get('right',            roop.globals.CFG.mask_right),
        mask_data.get('face_mask_blend',  roop.globals.CFG.face_mask_blend),
        mask_data.get('mouth_mask_blend', roop.globals.CFG.mouth_mask_blend),
        mask_data.get('mouth_top',        roop.globals.CFG.mouth_top_scale),
        mask_data.get('mouth_bottom',     roop.globals.CFG.mouth_bottom_scale),
        mask_data.get('mouth_left',       roop.globals.CFG.mouth_left_scale),
        mask_data.get('mouth_right',      roop.globals.CFG.mouth_right_scale),
    ]
    for fs in roop.globals.INPUT_FACESETS:
        for face in fs.faces:
            face.mask_offsets = list(offsets)


def _fe_reprocess_frames(orig_paths: list, orig_dir: str, meta: dict, fps: float):
    """Reprocess each original frame through live_swap with per-frame masks.

    Returns: (output_frames_dir: str, image_format: str) or (None, None) on failure.
    The caller is responsible for compiling the output frames.
    """
    import tempfile
    from roop.core import live_swap, get_processing_plugins
    from roop.ProcessOptions import ProcessOptions

    if not orig_paths or not roop.globals.INPUT_FACESETS:
        return None, None

    image_format = meta.get('image_format', roop.globals.CFG.output_image_format)

    # Build a stable ProcessOptions for this compile run (mask_json is per-frame)
    masking_engine = None  # no clip/text mask — per-frame canvas mask is passed via mask_json
    plugins  = get_processing_plugins(masking_engine)

    out_dir = tempfile.mkdtemp(prefix='fe_reprocess_')
    ok_count = 0
    for i, orig_path in enumerate(orig_paths):
        frame_bgr = cv2.imread(orig_path)
        if frame_bgr is None:
            print(f"[FrameEditor] could not read {orig_path}")
            continue

        basename  = os.path.basename(orig_path)
        mask_data = util.load_frame_mask(orig_dir, basename) if orig_dir else {}

        # Apply per-frame mask_offsets to facesets
        _fe_apply_mask_to_facesets(mask_data)

        canvas_mask_json = (mask_data.get('mask_json') or '').strip() or None

        options = ProcessOptions(
            plugins,
            roop.globals.distance_threshold,
            roop.globals.blend_ratio,
            roop.globals.face_swap_mode,
            0,                    # face_index
            None,                 # clip_text
            canvas_mask_json,
            1,                    # num_steps
            roop.globals.subsample_size if hasattr(roop.globals, 'subsample_size') else 128,
            roop.globals.CFG.show_mask_offsets,
            roop.globals.CFG.restore_original_mouth,
            use_3d_recon=False,
        )

        result = live_swap(frame_bgr, options)
        if result is None:
            result = frame_bgr

        out_path = os.path.join(out_dir, f"{i+1:06d}.{image_format}")
        cv2.imwrite(out_path, result)
        ok_count += 1

    if ok_count == 0:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        return None, None

    return out_dir, image_format


def on_fe_compile_mp4(frame_paths: list, orig_paths: list, orig_dir: str,
                       meta: dict, fps):
    """Reprocess original frames through face-swap with per-frame masks, compile MP4."""
    _no = (gr.update(visible=False), gr.update(visible=False))
    if not orig_paths and not frame_paths:
        return (*_no, "⚠️ No frames loaded.")
    if not orig_paths:
        return (*_no, "⚠️ No original frames found — run with 'Keep Frames' enabled.")
    if not roop.globals.INPUT_FACESETS:
        return (*_no, "⚠️ No source face loaded — load a source image in the Face Swap tab first.")

    fps_val      = float(fps) if fps else float(meta.get('fps', 24.0))
    source       = meta.get('source', 'output')
    source_base  = os.path.splitext(source)[0]
    output_path  = os.path.join(_fe_output_dir(frame_paths, orig_paths),
                                f"{source_base}_reprocessed.mp4")

    out_dir, image_format = _fe_reprocess_frames(orig_paths, orig_dir, meta, fps_val)
    if out_dir is None:
        return (*_no, "❌ Reprocessing failed — check the console for errors.")

    import shutil
    try:
        success = ffmpeg.create_video_from_frames_dir(out_dir, output_path, fps_val, image_format)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    if not success or not os.path.isfile(output_path):
        return (*_no, "❌ MP4 compilation failed — check the console for ffmpeg errors.")

    return (
        gr.update(visible=False),
        gr.update(visible=True, value=output_path),
        f"✅ Reprocessed → **{os.path.basename(output_path)}**",
    )


def on_fe_compile_gif(frame_paths: list, orig_paths: list, orig_dir: str,
                       meta: dict, fps):
    """Reprocess original frames through face-swap with per-frame masks, compile GIF."""
    _no = (gr.update(visible=False), gr.update(visible=False))
    if not orig_paths and not frame_paths:
        return (*_no, "⚠️ No frames loaded.")
    if not orig_paths:
        return (*_no, "⚠️ No original frames found — run with 'Keep Frames' enabled.")
    if not roop.globals.INPUT_FACESETS:
        return (*_no, "⚠️ No source face loaded — load a source image in the Face Swap tab first.")

    fps_val      = float(fps) if fps else float(meta.get('fps', 24.0))
    source       = meta.get('source', 'output')
    source_base  = os.path.splitext(source)[0]

    # Detect frame dimensions from first orig frame
    width = height = 0
    try:
        with Image.open(orig_paths[0]) as img:
            width, height = img.size
    except Exception:
        pass

    output_path = os.path.join(_fe_output_dir(frame_paths, orig_paths),
                               f"{source_base}_reprocessed.gif")

    out_dir, image_format = _fe_reprocess_frames(orig_paths, orig_dir, meta, fps_val)
    if out_dir is None:
        return (*_no, "❌ Reprocessing failed — check the console for errors.")

    import shutil
    try:
        success = ffmpeg.create_gif_from_frames_dir(
            out_dir, output_path, fps_val, width, height, image_format
        )
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    if not success or not os.path.isfile(output_path):
        return (*_no, "❌ GIF compilation failed — check the console for ffmpeg errors.")

    return (
        gr.update(visible=True, value=output_path),
        gr.update(visible=False),
        f"✅ Reprocessed → **{os.path.basename(output_path)}**",
    )
