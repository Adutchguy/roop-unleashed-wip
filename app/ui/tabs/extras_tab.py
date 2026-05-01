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
    # 🎞️ Frame Editor tab
    # ══════════════════════════════════════════════════════════════════════
    with gr.Tab("🎞️ Frame Editor", visible=True):
        # Persistent state for the loaded frame set
        fe_frames_list = gr.State([])   # sorted processed frame paths (_frames/)
        fe_orig_list   = gr.State([])   # sorted original (unswapped) frame paths (_frames_orig/)
        fe_orig_dir    = gr.State("")   # absolute path to _frames_orig/ directory
        fe_meta        = gr.State({})   # metadata dict (fps, source, image_format)

        # ── File drop loader ──────────────────────────────────────────
        fe_file_drop = gr.File(
            label="📂 Drop a video, GIF, or WebP file here to load its frames",
            file_count="single",
            file_types=["video", "image", ".webp", ".gif"],
        )

        with gr.Accordion("📁 Or load from an existing frames directory", open=False):
            with gr.Row():
                fe_dir_input = gr.Textbox(
                    label="Frames directory",
                    placeholder="Paste the path to a _frames folder, e.g. C:/output/myvideo_frames",
                    scale=5,
                )
                fe_load_btn = gr.Button("📂 Load", variant="primary", scale=1, min_width=80)

        fe_status = gr.Markdown("_Drop a media file above to load its frames, or expand the directory loader below._")

        # ── Frame navigation ─────────────────────────────────────────
        with gr.Row():
            fe_prev_btn = gr.Button("◀ Prev", size="sm", scale=1, min_width=80)
            fe_slider   = gr.Slider(minimum=1, maximum=1, value=1, step=1,
                                    label="Frame", scale=8)
            fe_next_btn = gr.Button("Next ▶", size="sm", scale=1, min_width=80)

        # ── Frame view (full-width with drawing support) ──────────────
        fe_frame_view = gr.ImageEditor(
            label="Current frame (draw to paint, eraser to undo strokes)",
            sources=None,           # background set programmatically
            type="numpy",
            image_mode="RGB",
            height=520,
            layers=True,            # separate drawing layer for face-space tracking
            transforms=(),          # disable crop/resize controls
            brush=gr.Brush(
                colors=["#ff0000", "#00ff00", "#0000ff",
                        "#ffffff", "#000000", "#ffff00", "#ff8800"],
                default_size=8,
                color_mode="defaults",
            ),
            eraser=gr.Eraser(default_size=20),
            value=None,
        )

        # ── Drawing save / revert ────────────────────────────────────
        with gr.Row():
            fe_draw_save_btn   = gr.Button("💾 Save Drawing to Frame",
                                           variant="primary", scale=2)
            fe_draw_revert_btn = gr.Button("↩ Revert Frame",
                                           variant="secondary", scale=1)

        # ── Apply drawing across a frame range ───────────────────────
        with gr.Group():
            gr.Markdown("#### Apply Drawing to Frame Range")
            with gr.Row():
                fe_range_start = gr.Number(value=1, label="From frame",
                                           minimum=1, step=1, scale=1)
                fe_range_end   = gr.Number(value=1, label="To frame",
                                           minimum=1, step=1, scale=1)
            with gr.Row():
                fe_apply_tracked_btn = gr.Button(
                    "🎯 Face Tracking", variant="primary", scale=2
                )
                fe_apply_person_btn  = gr.Button(
                    "🏃 Body Tracking", variant="primary", scale=2
                )
                fe_apply_range_btn   = gr.Button(
                    "📋 Flat (no tracking)", variant="secondary", scale=1
                )

        fe_draw_status = gr.Markdown("")

        # ── Hidden mask components (kept for JS bridge compatibility) ─
        with gr.Column(visible=False):
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
            fe_mouth_top = gr.Slider(
                0, 10.0, value=roop.globals.CFG.mouth_top_scale,
                label="Mouth Top", step=0.1, interactive=True,
            )
            fe_mouth_bottom = gr.Slider(
                0, 10.0, value=roop.globals.CFG.mouth_bottom_scale,
                label="Mouth Bottom", step=0.1, interactive=True,
            )
            fe_mouth_left = gr.Slider(
                0, 10.0, value=roop.globals.CFG.mouth_left_scale,
                label="Mouth Left", step=0.1, interactive=True,
            )
            fe_mouth_right = gr.Slider(
                0, 10.0, value=roop.globals.CFG.mouth_right_scale,
                label="Mouth Right", step=0.1, interactive=True,
            )
            fe_mask_btn      = gr.Button("🎭 Edit Canvas Mask", variant="secondary")
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

        # ── Compile current frames (simple stitch — no face swap) ──────
        with gr.Group():
            gr.Markdown("#### Compile Frames")
            with gr.Row(variant="panel"):
                fe_fps = gr.Number(value=24.0, label="Output FPS",
                                   minimum=1, maximum=120, scale=1, min_width=120)
                fe_compile_current_mp4_btn = gr.Button("🎬 Compile → MP4", variant="primary", scale=2)
                fe_compile_current_gif_btn = gr.Button("🎞️ Compile → GIF", variant="primary", scale=2)
            gr.Markdown(
                "_Stitches the current frame images (with any drawings baked in) directly "
                "into a video or GIF.  No face-swap is applied._"
            )

        # ── Reprocess (face-swap pipeline from originals) ───────────
        with gr.Group():
            gr.Markdown("#### Reprocess (Face Swap)")
            with gr.Row(variant="panel"):
                fe_compile_mp4_btn = gr.Button("🎬 Reprocess → MP4", scale=2)
                fe_compile_gif_btn = gr.Button("🎞️ Reprocess → GIF", scale=2)
            gr.Markdown(
                "_Reruns the face-swap pipeline on each **original** (unswapped) frame "
                "using saved per-frame mask settings, then compiles.  "
                "Requires originals and a loaded source face._"
            )

        # ── Compiled output preview ───────────────────────────────────
        with gr.Row():
            fe_out_image = gr.Image(label="Output",
                                    visible=False, interactive=False,
                                    show_download_button=True)
            fe_out_video = gr.Video(label="Output",
                                    visible=False, interactive=False)

    # ── All slider components for mask I/O ───────────────────────────
    _fe_mask_sliders = [
        fe_mask_top, fe_mask_bottom, fe_mask_left, fe_mask_right,
        fe_face_blend,
        fe_mouth_blend, fe_mouth_top, fe_mouth_bottom,
        fe_mouth_left, fe_mouth_right,
    ]

    # ── Frame Editor event wiring ─────────────────────────────────────

    # Helper to sync range-end to total frame count after loading
    def _fe_sync_range_end(frames):
        n = len(frames)
        return gr.update(value=n) if n > 0 else gr.update()

    # File drop — clear resets the entire Frame Editor
    fe_file_drop.clear(
        fn=on_fe_clear,
        outputs=[
            fe_slider, fe_status,
            fe_frames_list, fe_orig_list, fe_orig_dir, fe_meta,
            fe_fps, fe_range_start, fe_range_end,
            fe_frame_view, fe_draw_status,
            fe_mask_json_store, fe_mask_face_crop_store, fe_mask_face_swap_crop_store,
            fe_out_image, fe_out_video,
            *_fe_mask_sliders,
        ],
        show_progress="hidden",
    )

    # File drop loader (primary)
    fe_file_drop.upload(
        fn=on_fe_load_file,
        inputs=[fe_file_drop],
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
    ).then(
        fn=_fe_sync_range_end,
        inputs=[fe_frames_list],
        outputs=[fe_range_end],
        show_progress="hidden",
    )

    # Directory load (fallback, inside accordion)
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
    ).then(
        fn=_fe_sync_range_end,
        inputs=[fe_frames_list],
        outputs=[fe_range_end],
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

    # Save drawing to frame / revert frame
    fe_draw_save_btn.click(
        fn=on_fe_save_drawing,
        inputs=[fe_frame_view, fe_slider, fe_frames_list],
        outputs=[fe_draw_status],
    )
    fe_draw_revert_btn.click(
        fn=on_fe_revert_frame,
        inputs=[fe_slider, fe_frames_list, fe_orig_list],
        outputs=[fe_frame_view, fe_draw_status],
    )

    # Apply drawing with face tracking across a range
    fe_apply_tracked_btn.click(
        fn=on_fe_apply_tracked,
        inputs=[fe_frame_view, fe_slider, fe_range_start, fe_range_end, fe_frames_list],
        outputs=[fe_draw_status],
    )

    # Apply drawing with optical-flow body tracking across a range
    fe_apply_person_btn.click(
        fn=on_fe_apply_person_tracked,
        inputs=[fe_frame_view, fe_slider, fe_range_start, fe_range_end, fe_frames_list],
        outputs=[fe_draw_status],
    )

    # Apply drawing flat (no tracking) across a range
    fe_apply_range_btn.click(
        fn=on_fe_apply_range,
        inputs=[fe_frame_view, fe_slider, fe_range_start, fe_range_end, fe_frames_list],
        outputs=[fe_draw_status],
    )

    # Compile current frames (simple stitch — no face swap)
    fe_compile_current_mp4_btn.click(
        fn=on_fe_compile_current_mp4,
        inputs=[fe_frames_list, fe_fps, fe_meta],
        outputs=[fe_out_image, fe_out_video, fe_status],
    )
    fe_compile_current_gif_btn.click(
        fn=on_fe_compile_current_gif,
        inputs=[fe_frames_list, fe_fps, fe_meta],
        outputs=[fe_out_image, fe_out_video, fe_status],
    )

    # Reprocess (face-swap pipeline from originals)
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
            # Animated webp → GIF output (consistent with swap pipeline).
            # PIL pipes frames through ffmpeg to a temp mp4, then we convert
            # that to a palette-optimised GIF and discard the mp4.
            base = os.path.splitext(dest)[0]
            dest_mp4 = base + '__temp.mp4'
            dest_gif = base + '.gif'
            success = ffmpeg.apply_media_transforms_webp(f, dest_mp4, filters, cur_fps)
            if success and os.path.isfile(dest_mp4):
                # Pass cur_fps explicitly so the GIF uses the original WebP timing
                # rather than re-detecting it from the intermediate MP4.
                ffmpeg.create_gif_from_video(dest_mp4, dest_gif, target_fps=cur_fps)
                os.remove(dest_mp4)
                success = os.path.isfile(dest_gif)
            dest = dest_gif
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


def on_fe_save_drawing(editor_value, frame_num, frame_paths: list):
    """Composite the ImageEditor drawing onto the frame file on disk.

    The ImageEditor returns a dict with keys 'background', 'layers', and
    'composite'. The 'composite' is the merged RGBA numpy array; we flatten
    it to RGB (alpha channel blended over the original file) and overwrite
    the processed frame so it is picked up when reprocessing or compiling.
    """
    if not frame_paths:
        return "⚠️ No frames loaded."
    if not isinstance(editor_value, dict):
        return "⚠️ No editor value received."

    composite = editor_value.get("composite")
    if composite is None:
        return "ℹ️ Nothing drawn yet — make a stroke first."

    idx = max(0, int(frame_num) - 1)
    if idx >= len(frame_paths):
        return "⚠️ Frame index out of range."
    frame_path = frame_paths[idx]
    if not os.path.isfile(frame_path):
        return f"⚠️ Frame file not found: {os.path.basename(frame_path)}"

    # composite arrives as (H, W, 4) RGBA or (H, W, 3) RGB numpy array
    if composite.ndim == 3 and composite.shape[2] == 4:
        alpha  = composite[:, :, 3:4].astype(np.float32) / 255.0
        rgb_fg = composite[:, :, :3].astype(np.float32)
        # Blend drawing over the on-disk frame using the alpha channel
        orig_bgr = cv2.imread(frame_path)
        if orig_bgr is None:
            return f"⚠️ Could not read {os.path.basename(frame_path)} from disk."
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        blended  = (rgb_fg * alpha + orig_rgb * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    elif composite.ndim == 3 and composite.shape[2] == 3:
        bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    else:
        return "⚠️ Unexpected composite format — could not save."

    cv2.imwrite(frame_path, bgr)
    return f"✅ Drawing saved to frame {int(frame_num)}"


def on_fe_revert_frame(frame_num, frame_paths: list, orig_paths: list):
    """Copy the original (unswapped) frame back over the processed frame on disk,
    then reload the ImageEditor so the canvas reflects the reverted state.
    """
    import shutil as _shutil
    no_update = gr.update()
    if not frame_paths:
        return no_update, "⚠️ No frames loaded."
    if not orig_paths:
        return no_update, "⚠️ No originals found — run swap with 'Keep Frames' enabled."

    idx = max(0, int(frame_num) - 1)
    if idx >= len(orig_paths) or idx >= len(frame_paths):
        return no_update, "⚠️ Frame index out of range."

    orig_path = orig_paths[idx]
    proc_path = frame_paths[idx]
    if not os.path.isfile(orig_path):
        return no_update, f"⚠️ Original not found: {os.path.basename(orig_path)}"

    _shutil.copy2(orig_path, proc_path)
    editor_value = {"background": proc_path, "layers": [], "composite": None}
    return gr.update(value=editor_value), f"↩ Frame {int(frame_num)} reverted to original."


def on_fe_clear():
    """Reset all Frame Editor state when the source file is cleared."""
    cfg = roop.globals.CFG
    default_sliders = [
        cfg.mask_top, cfg.mask_bottom, cfg.mask_left, cfg.mask_right,
        cfg.face_mask_blend,
        cfg.mouth_mask_blend, cfg.mouth_top_scale, cfg.mouth_bottom_scale,
        cfg.mouth_left_scale, cfg.mouth_right_scale,
    ]
    return (
        gr.update(value=1, minimum=1, maximum=1),                   # fe_slider
        "_Drop a media file above to load its frames, or expand the directory loader below._",  # fe_status
        [],                                                          # fe_frames_list
        [],                                                          # fe_orig_list
        "",                                                          # fe_orig_dir
        {},                                                          # fe_meta
        gr.update(value=24.0),                                       # fe_fps
        gr.update(value=1),                                          # fe_range_start
        gr.update(value=1),                                          # fe_range_end
        gr.update(value=None),                                       # fe_frame_view
        "",                                                          # fe_draw_status
        gr.update(value=""),                                         # fe_mask_json_store
        gr.update(value=""),                                         # fe_mask_face_crop_store
        gr.update(value=""),                                         # fe_mask_face_swap_crop_store
        gr.update(visible=False, value=None),                        # fe_out_image
        gr.update(visible=False, value=None),                        # fe_out_video
        *[gr.update(value=v) for v in default_sliders],             # _fe_mask_sliders (10)
    )


def on_fe_load_file(file):
    """Extract frames from a dropped media file and load them into the Frame Editor.

    Accepts a gr.File upload object (video, animated GIF, animated WebP, or static image).
    Frames are extracted to the standard roop temp directory for that file, then
    on_fe_load() is called with that directory so all existing state logic is reused.
    """
    _empty = (
        gr.update(value=1, minimum=1, maximum=1),
        "_No frames loaded._",
        [], [], "", {},
        gr.update(value=24.0),
    )
    if file is None:
        return _empty

    file_path = file.name if hasattr(file, 'name') else str(file)
    if not os.path.isfile(file_path):
        return _empty

    is_awebp    = util.is_animated_webp(file_path)
    is_agif     = util.is_animated_gif(file_path)
    is_animated = is_awebp or is_agif
    is_img      = util.is_image(file_path) and not is_animated

    # Static image: just browse its parent directory
    if is_img:
        return on_fe_load(os.path.dirname(file_path))

    # Detect fps for the frame extraction command
    try:
        fps = util.detect_fps(file_path)
    except Exception:
        fps = 24.0

    gr_status = f"_Extracting frames from **{os.path.basename(file_path)}** at {fps:.2f} fps…_"
    # Extract frames into the standard temp directory
    ok = ffmpeg.extract_frames(file_path, None, None, fps)
    if not ok:
        return (
            gr.update(value=1, minimum=1, maximum=1),
            f"⚠️ Frame extraction failed for **{os.path.basename(file_path)}**.",
            [], [], "", {},
            gr.update(value=fps),
        )

    temp_dir     = util.get_temp_directory_path(file_path)
    image_format = roop.globals.CFG.output_image_format
    # Write meta.json so on_fe_load() reads the correct fps instead of defaulting to 24
    util.write_frames_metadata(temp_dir, fps, os.path.basename(file_path), image_format)
    return on_fe_load(temp_dir)


def _fe_extract_drawing(editor_value: dict, src_bgr: np.ndarray):
    """Return (draw_rgb, draw_mask) containing ONLY the drawn strokes.

    draw_rgb  — float32 (H, W, 3)  brush colour at stroke pixels, 0 elsewhere
    draw_mask — float32 (H, W, 1)  stroke alpha, feathered; 0.0 where nothing drawn

    Returns (None, None) if nothing was drawn.

    CRITICAL: draw_rgb must be zero outside stroke pixels.  If it contains
    source-frame content at near-zero-alpha positions, that content bleeds
    into target frames as a faint (or not-so-faint) ghost of the original
    frame, looking like a copy of the frame being waved around.

    Source priority
    ---------------
    1. layers[0] RGBA alpha — the true drawing-layer alpha; brush colour only.
    2. background-vs-composite diff — both images come from the same browser
       round-trip so JPEG encoding is identical outside strokes → diff=0 there.
       Far more accurate than diffing against the on-disk file (which has
       different compression and creates false positives across the whole frame).
    3. composite-vs-on-disk diff — least reliable, kept only as last resort.
    """
    h_src, w_src = src_bgr.shape[:2]

    def _rgb3(img):
        """Return float32 RGB (H,W,3), dropping alpha if present."""
        img = img.astype(np.float32)
        return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img

    def _fit(img):
        """Resize to (h_src, w_src) if needed."""
        if img.shape[:2] != (h_src, w_src):
            return cv2.resize(img, (w_src, h_src), interpolation=cv2.INTER_LINEAR)
        return img

    def _mask_from_binary(binary, stroke_rgb):
        """Feather a uint8 binary mask and zero draw_rgb outside stroke pixels.

        Zeroing draw_rgb prevents source-frame content from bleeding through
        at feathered edge pixels that have small but non-zero alpha.
        """
        alpha_f   = cv2.GaussianBlur(
            binary.astype(np.float32), (0, 0), sigmaX=4
        ).clip(0.0, 1.0)
        # draw_rgb = stroke colour where drawn, 0 elsewhere
        draw_rgb  = np.where(binary[:, :, np.newaxis].astype(bool),
                             stroke_rgb, np.float32(0))
        return draw_rgb, alpha_f[:, :, np.newaxis]

    # ── 1. Drawing layer RGBA alpha ──────────────────────────────────────
    for layer in (editor_value.get("layers") or []):
        if layer is None:
            continue
        layer = np.asarray(layer)   # handles PIL Images and numpy arrays alike
        if layer.ndim != 3 or layer.shape[2] != 4:
            continue
        layer   = _fit(layer)
        alpha_f = layer[:, :, 3].astype(np.float32) / 255.0
        if alpha_f.max() > 0.02:
            # Layer RGB is the raw brush colour — zero background bleed possible
            alpha_soft = cv2.GaussianBlur(alpha_f, (0, 0), sigmaX=4).clip(0.0, 1.0)
            draw_rgb   = np.where(
                (alpha_f > 0.02)[:, :, np.newaxis],
                layer[:, :, :3].astype(np.float32),
                np.float32(0),
            )
            return draw_rgb, alpha_soft[:, :, np.newaxis]

    composite  = editor_value.get("composite")
    background = editor_value.get("background")

    # ── 2. background-vs-composite diff (same browser encoding) ─────────
    if composite is not None and background is not None:
        comp_rgb = _rgb3(_fit(np.asarray(composite)))
        bg_rgb   = _rgb3(_fit(np.asarray(background)))
        diff     = np.abs(comp_rgb - bg_rgb).max(axis=2)
        # Low threshold: both arrays share the same JPEG encoding artefacts
        # so diff outside strokes should be ≈0
        binary   = (diff > 5).astype(np.uint8)
        kernel   = np.ones((3, 3), np.uint8)
        binary   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        if binary.sum() > 0:
            return _mask_from_binary(binary, comp_rgb)

    # ── 3. composite-vs-on-disk diff (last resort) ──────────────────────
    if composite is None:
        return None, None
    comp_rgb = _rgb3(_fit(np.asarray(composite)))
    src_rgb  = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    diff     = np.abs(comp_rgb - src_rgb).max(axis=2)
    binary   = (diff > 10).astype(np.uint8)
    kernel   = np.ones((3, 3), np.uint8)
    binary   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    if binary.sum() < 1:
        return None, None
    return _mask_from_binary(binary, comp_rgb)


def _drawing_crop(draw_rgb, draw_mask, h_src, w_src, pad: int = 30):
    """Return (crop_rgba, x1_c, y1_c) — tight bbox crop of the drawing + padding.

    Warping only this small patch (instead of the full-frame drawing) eliminates
    the large warp-boundary rectangle that previously appeared around tracked regions.
    """
    mask_sq = draw_mask.squeeze()
    ys, xs  = np.where(mask_sq > 0.05)
    if len(ys) == 0:
        return None, 0, 0
    y1_c = max(0, int(ys.min()) - pad)
    y2_c = min(h_src, int(ys.max()) + pad)
    x1_c = max(0, int(xs.min()) - pad)
    x2_c = min(w_src, int(xs.max()) + pad)
    alpha_u8 = (draw_mask * 255).clip(0, 255).astype(np.uint8)
    rgb_u8   = draw_rgb.clip(0, 255).astype(np.uint8)
    rgba     = np.concatenate([rgb_u8, alpha_u8], axis=2)
    return rgba[y1_c:y2_c, x1_c:x2_c], x1_c, y1_c


def _blend_patch(tgt_bgr: np.ndarray,
                 crop_rgba: np.ndarray,
                 x1_c: int, y1_c: int,
                 M_dst_to_src: np.ndarray) -> np.ndarray:
    """Warp crop_rgba into tgt_bgr's space using M_dst_to_src and alpha-blend.

    M_dst_to_src is a 2×3 affine in OpenCV's *inverse-map* convention:
        for each destination pixel (x′,y′), the source pixel is M · [x′,y′,1]ᵀ.

    Since the source is crop_rgba (not the full-frame drawing), we shift the
    translation column by (-x1_c, -y1_c) so that lookups hit the crop's
    coordinate system rather than full-frame coordinates.
    """
    h, w   = tgt_bgr.shape[:2]
    M_crop = M_dst_to_src.copy().astype(np.float32)
    M_crop[0, 2] -= x1_c
    M_crop[1, 2] -= y1_c
    warped = cv2.warpAffine(crop_rgba, M_crop, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    alpha   = warped[:, :, 3:4].astype(np.float32) / 255.0
    rgb_fg  = warped[:, :, :3].astype(np.float32)
    tgt_rgb = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    blended = (rgb_fg * alpha + tgt_rgb * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


def on_fe_apply_tracked(editor_value, frame_num, range_start, range_end, frame_paths: list):
    """Warp the drawn strokes — tracked to the face — onto every frame in the range.

    OpenCV warpAffine convention (inverse map):
        dst(x′, y′) = src( M · [x′, y′, 1]ᵀ )
    So every M we pass to warpAffine must map *destination* coords → *source* coords.

    estimate_norm(kps, 112) returns M such that warpAffine(frame, M, (112,112))
    gives the aligned face crop → M maps face-space coords to frame coords.

    Pipeline for each target frame:
        target pixel t
            ──[M_tgt_inv = invertAffineTransform(M_tgt)]──▶ face-space point f
            ──[M_src (maps face-space → source frame)]──▶  source drawing pixel s

    Combined (single warpAffine): M_direct = M_src_3×3 @ M_tgt_inv_3×3
    """
    from roop.face_util import get_first_face, estimate_norm

    if not frame_paths:
        return "⚠️ No frames loaded."
    if not isinstance(editor_value, dict):
        return "⚠️ No editor value — open a frame first."

    cur_idx = max(0, int(frame_num) - 1)
    if cur_idx >= len(frame_paths):
        return "⚠️ Frame index out of range."
    src_bgr = cv2.imread(frame_paths[cur_idx])
    if src_bgr is None:
        return "⚠️ Could not read source frame from disk."

    draw_rgb, draw_mask = _fe_extract_drawing(editor_value, src_bgr)
    if draw_rgb is None:
        return "ℹ️ No drawing detected — make a stroke on the current frame first."

    h_src, w_src = src_bgr.shape[:2]
    crop_rgba, x1_c, y1_c = _drawing_crop(draw_rgb, draw_mask, h_src, w_src)
    if crop_rgba is None:
        return "ℹ️ No drawing detected."

    src_face = get_first_face(src_bgr)
    if src_face is None:
        return "⚠️ No face detected in current frame — cannot establish face-space transform."

    face_size = 112
    # M_src: face-space coords → source frame coords  (inverse-map convention)
    M_src     = estimate_norm(src_face.kps, face_size)
    M_src_3x3 = np.vstack([M_src, [0.0, 0.0, 1.0]])

    # Fallback inverse: identity warp in source-frame space
    M_fallback = np.float32([[1, 0, 0], [0, 1, 0]])

    start_idx = max(0, int(range_start) - 1)
    end_idx   = min(len(frame_paths) - 1, int(range_end) - 1)
    if start_idx > end_idx:
        return "⚠️ Invalid range (From frame > To frame)."

    applied = 0
    no_face = 0
    for i in range(start_idx, end_idx + 1):
        tgt_path = frame_paths[i]
        tgt_bgr  = cv2.imread(tgt_path)
        if tgt_bgr is None:
            continue

        h_tgt, w_tgt = tgt_bgr.shape[:2]
        tgt_face = get_first_face(tgt_bgr)

        if tgt_face is not None:
            M_tgt     = estimate_norm(tgt_face.kps, face_size)
            # M_tgt_inv: target frame coords → face-space coords
            M_tgt_inv = cv2.invertAffineTransform(M_tgt)
            M_tgt_inv_3x3 = np.vstack([M_tgt_inv, [0.0, 0.0, 1.0]])
            # M_direct: target pixel → face-space → source frame pixel  (dst→src)
            M_direct = (M_src_3x3 @ M_tgt_inv_3x3)[:2]
        else:
            # No face in this frame — keep drawing at the same screen position
            M_direct = M_fallback
            no_face += 1

        result = _blend_patch(tgt_bgr, crop_rgba, x1_c, y1_c, M_direct)
        cv2.imwrite(tgt_path, result)
        applied += 1

    msg = f"✅ Drawing applied (face-tracked) to **{applied}** frame(s)"
    if no_face:
        msg += f" — ⚠️ {no_face} frame(s) held position (no face detected)"
    return msg


def on_fe_apply_person_tracked(editor_value, frame_num, range_start, range_end,
                                frame_paths: list):
    """Track the drawn strokes across frames using sparse Lucas-Kanade optical flow.

    Key correctness notes
    ---------------------
    estimateAffinePartial2D(src_pts, dst_pts) returns M_fwd such that:
        dst_pt ≈ M_fwd · [src_pt_x, src_pt_y, 1]ᵀ   (forward map: source → current)

    warpAffine(src_img, M, dsize) uses *inverse* mapping:
        dst(x′, y′) = src_img( M · [x′,y′,1]ᵀ )      (M must map dst → src)

    Therefore we must invert M_fwd before passing it to warpAffine so that each
    current-frame pixel correctly looks up the original source drawing pixel.

    We also warp only a tight crop around the drawn strokes (not the full frame)
    so there is no large warped-rectangle boundary visible in the result.
    """
    if not frame_paths:
        return "⚠️ No frames loaded."
    if not isinstance(editor_value, dict):
        return "⚠️ No editor value — open a frame first."

    cur_idx   = max(0, int(frame_num) - 1)
    start_idx = max(0, int(range_start) - 1)
    end_idx   = min(len(frame_paths) - 1, int(range_end) - 1)
    if cur_idx >= len(frame_paths):
        return "⚠️ Frame index out of range."
    if start_idx > end_idx:
        return "⚠️ Invalid range (From frame > To frame)."

    src_bgr = cv2.imread(frame_paths[cur_idx])
    if src_bgr is None:
        return "⚠️ Could not read source frame."

    draw_rgb, draw_mask = _fe_extract_drawing(editor_value, src_bgr)
    if draw_rgb is None:
        return "ℹ️ No drawing detected — make a stroke on the current frame first."

    h_fr, w_fr = src_bgr.shape[:2]

    # Crop the drawing tightly around the strokes — warping only this small patch
    # prevents the full-frame warp boundary from appearing as a visible rectangle.
    crop_rgba, x1_c, y1_c = _drawing_crop(draw_rgb, draw_mask, h_fr, w_fr)
    if crop_rgba is None:
        return "ℹ️ No drawing detected."

    # ── Sample feature points around the drawn region ─────────────────
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    mask_sq  = draw_mask.squeeze()
    ys, xs   = np.where(mask_sq > 0.05)

    # Expand bbox ~40 px for richer Shi-Tomasi corners
    y1_roi = max(0, int(ys.min()) - 40);  y2_roi = min(h_fr, int(ys.max()) + 40)
    x1_roi = max(0, int(xs.min()) - 40);  x2_roi = min(w_fr, int(xs.max()) + 40)
    roi_mask = np.zeros_like(src_gray)
    roi_mask[y1_roi:y2_roi, x1_roi:x2_roi] = 255

    pts = cv2.goodFeaturesToTrack(
        src_gray, maxCorners=300, qualityLevel=0.005,
        minDistance=4, mask=roi_mask,
    )
    if pts is None or len(pts) < 4:
        idx = np.random.choice(len(ys), min(300, len(ys)), replace=False)
        pts = np.stack([xs[idx], ys[idx]], axis=1).reshape(-1, 1, 2).astype(np.float32)

    initial_pts = pts.reshape(-1, 2)   # (N,2) — fixed reference positions in source frame

    LK = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # ── Helper: LK step → (M_fwd, tracked_pts, valid_count) ─────────
    def _lk_step(prev_gray, curr_gray, cur_pts):
        """Track cur_pts into curr_gray; return (M_fwd, next_pts, n_valid).

        M_fwd maps initial_pts → next_pts (forward: source → current frame).
        Caller must invertAffineTransform(M_fwd) before warpAffine.
        """
        next_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            cur_pts.reshape(-1, 1, 2).astype(np.float32),
            None, **LK,
        )
        valid = st.reshape(-1).astype(bool)
        M_fwd = None
        if valid.sum() >= 3:
            M_fwd, _ = cv2.estimateAffinePartial2D(
                initial_pts[valid].reshape(-1, 1, 2),
                next_pts.reshape(-1, 2)[valid].reshape(-1, 1, 2),
            )
        next_pts_flat = next_pts.reshape(-1, 2)
        next_pts_flat[~valid] = cur_pts.reshape(-1, 2)[~valid]
        return M_fwd, next_pts_flat, int(valid.sum())

    # ── Helper: apply tracked warp to one frame ───────────────────────
    def _apply_fwd(tgt_bgr, M_fwd):
        """Invert M_fwd (src→dst) to get dst→src for warpAffine, then blend crop."""
        if M_fwd is None:
            return tgt_bgr
        M_inv = cv2.invertAffineTransform(M_fwd)   # now maps current → source ✓
        return _blend_patch(tgt_bgr, crop_rgba, x1_c, y1_c, M_inv)

    applied = 0

    # Source frame: drawing sits at its natural position (identity inverse map)
    if start_idx <= cur_idx <= end_idx:
        tgt_bgr = cv2.imread(frame_paths[cur_idx])
        if tgt_bgr is not None:
            M_id = np.float32([[1, 0, 0], [0, 1, 0]])   # identity: dst→src is identity
            cv2.imwrite(frame_paths[cur_idx],
                        _blend_patch(tgt_bgr, crop_rgba, x1_c, y1_c, M_id))
            applied += 1

    # ── Forward pass: cur_idx+1 → end_idx ────────────────────────────
    prev_gray = src_gray
    fwd_pts   = initial_pts.copy()
    for i in range(cur_idx + 1, end_idx + 1):
        curr_bgr = cv2.imread(frame_paths[i])
        if curr_bgr is None:
            continue
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        M_fwd, fwd_pts, _ = _lk_step(prev_gray, curr_gray, fwd_pts)
        if M_fwd is not None:
            cv2.imwrite(frame_paths[i], _apply_fwd(curr_bgr, M_fwd))
            applied += 1
        prev_gray = curr_gray

    # ── Backward pass: cur_idx-1 → start_idx ─────────────────────────
    prev_gray = src_gray
    bwd_pts   = initial_pts.copy()
    for i in range(cur_idx - 1, start_idx - 1, -1):
        curr_bgr = cv2.imread(frame_paths[i])
        if curr_bgr is None:
            continue
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        M_fwd, bwd_pts, _ = _lk_step(prev_gray, curr_gray, bwd_pts)
        if M_fwd is not None:
            cv2.imwrite(frame_paths[i], _apply_fwd(curr_bgr, M_fwd))
            applied += 1
        prev_gray = curr_gray

    return f"✅ Drawing applied (body-tracked) to **{applied}** frame(s)"


def on_fe_apply_range(editor_value, frame_num, range_start, range_end, frame_paths: list):
    """Paste the drawing flat (no face tracking) onto every frame in the range.

    Uses the same diff-vs-source approach as on_fe_apply_tracked to isolate
    the drawn pixels reliably regardless of Gradio's layers API behaviour.
    """
    if not frame_paths:
        return "⚠️ No frames loaded."
    if not isinstance(editor_value, dict):
        return "⚠️ No editor value — open a frame first."

    # Read source frame to compute drawing via diff
    cur_idx = max(0, int(frame_num) - 1)
    if cur_idx >= len(frame_paths):
        return "⚠️ Frame index out of range."
    src_bgr = cv2.imread(frame_paths[cur_idx])
    if src_bgr is None:
        return "⚠️ Could not read source frame from disk."

    draw_rgb, draw_mask = _fe_extract_drawing(editor_value, src_bgr)
    if draw_rgb is None:
        return "ℹ️ No drawing detected — make a stroke on the current frame first."

    start_idx = max(0, int(range_start) - 1)
    end_idx   = min(len(frame_paths) - 1, int(range_end) - 1)
    if start_idx > end_idx:
        return "⚠️ Invalid range (From frame > To frame)."

    h_src, w_src = src_bgr.shape[:2]
    applied = 0
    for i in range(start_idx, end_idx + 1):
        tgt_path = frame_paths[i]
        tgt_bgr  = cv2.imread(tgt_path)
        if tgt_bgr is None:
            continue

        h, w = tgt_bgr.shape[:2]
        if (h, w) != (h_src, w_src):
            _rgb  = cv2.resize(draw_rgb.astype(np.uint8), (w, h)).astype(np.float32)
            _mask = cv2.resize(draw_mask.squeeze(), (w, h))[:, :, np.newaxis]
        else:
            _rgb, _mask = draw_rgb, draw_mask

        tgt_rgb = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        blended = (_rgb * _mask + tgt_rgb * (1.0 - _mask)).clip(0, 255).astype(np.uint8)
        cv2.imwrite(tgt_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        applied += 1

    return f"✅ Drawing applied (flat) to **{applied}** frame(s)"


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

    # ImageEditor expects a dict with a 'background' key to set the canvas
    # without disturbing any in-progress drawing session.  Passing a plain
    # filepath string also works and clears any existing drawing layers.
    editor_value = {"background": proc_path, "layers": [], "composite": None} if proc_path else None
    return (
        gr.update(value=editor_value),
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


def on_fe_compile_current_mp4(frame_paths: list, fps, meta: dict):
    """Stitch the current processed frame images directly into an MP4 (no face swap)."""
    _no = (gr.update(visible=False), gr.update(visible=False))
    if not frame_paths:
        return (*_no, "⚠️ No frames loaded.")

    fps_val      = float(fps) if fps else float(meta.get('fps', 24.0))
    image_format = meta.get('image_format', roop.globals.CFG.output_image_format)
    frames_dir   = os.path.dirname(frame_paths[0])
    source       = meta.get('source', 'output')
    source_base  = os.path.splitext(os.path.basename(source))[0] if source else 'output'
    output_path  = os.path.join(_fe_output_dir(frame_paths, []),
                                f"{source_base}_compiled.mp4")

    success = ffmpeg.create_video_from_frames_dir(frames_dir, output_path, fps_val, image_format)
    if not success or not os.path.isfile(output_path):
        return (*_no, "❌ MP4 compilation failed — check the console for ffmpeg errors.")

    return (
        gr.update(visible=False),
        gr.update(visible=True, value=output_path),
        f"✅ Compiled → **{os.path.basename(output_path)}**",
    )


def on_fe_compile_current_gif(frame_paths: list, fps, meta: dict):
    """Stitch the current processed frame images directly into an animated GIF (no face swap)."""
    _no = (gr.update(visible=False), gr.update(visible=False))
    if not frame_paths:
        return (*_no, "⚠️ No frames loaded.")

    fps_val      = float(fps) if fps else float(meta.get('fps', 24.0))
    image_format = meta.get('image_format', roop.globals.CFG.output_image_format)
    frames_dir   = os.path.dirname(frame_paths[0])
    source       = meta.get('source', 'output')
    source_base  = os.path.splitext(os.path.basename(source))[0] if source else 'output'
    output_path  = os.path.join(_fe_output_dir(frame_paths, []),
                                f"{source_base}_compiled.gif")

    # Detect frame dimensions from first frame
    width = height = 0
    try:
        with Image.open(frame_paths[0]) as img:
            width, height = img.size
    except Exception:
        pass

    success = ffmpeg.create_gif_from_frames_dir(
        frames_dir, output_path, fps_val, width, height, image_format
    )
    if not success or not os.path.isfile(output_path):
        return (*_no, "❌ GIF compilation failed — check the console for ffmpeg errors.")

    return (
        gr.update(visible=True, value=output_path),
        gr.update(visible=False),
        f"✅ Compiled → **{os.path.basename(output_path)}**",
    )


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
