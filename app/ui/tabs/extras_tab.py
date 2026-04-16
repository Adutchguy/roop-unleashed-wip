import os
import gradio as gr
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
