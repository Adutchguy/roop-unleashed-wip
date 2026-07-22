import os

import gradio as gr

import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import roop.globals

# gradio_rangeslider gives us a single dual-handle timeline for trim start/end.
# It's an optional extra (see requirements.txt) — if it isn't installed yet
# (e.g. the launcher hasn't been reinstalled since it was added), fall back to
# a plain pair of Start/End sliders so the tab still works.
try:
    from gradio_rangeslider import RangeSlider
    HAS_RANGESLIDER = True
except Exception:
    RangeSlider = None
    HAS_RANGESLIDER = False

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

_EMPTY_INFO = {"width": 0, "height": 0, "fps": 24.0, "duration": 0.0, "is_video": False}


def extras_tab(bt_destfiles=None):
    # State: tracks detected properties of the current file
    file_info = gr.State(dict(_EMPTY_INFO))

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

        # ── Trim ──────────────────────────────────────────────────────
        with gr.Group(visible=False) as trim_group:
            if HAS_RANGESLIDER:
                gr.Markdown("#### Trim  *(drag either handle to set the start / end time)*")
            else:
                gr.Markdown("#### Trim  *(set a start / end time, in seconds)*")
            current_duration_label = gr.Markdown("**Current:** —")
            with gr.Row():
                # Fixed-size thumbnail boxes — the actual frame (whatever its
                # aspect ratio) is scaled down to fit inside via object-fit:
                # contain, rather than stretching the layout to the frame size.
                trim_start_preview = gr.Image(
                    label="Start frame", visible=False, interactive=False,
                    show_download_button=False,
                    height=120, width=160, scale=0,
                )
                trim_end_preview = gr.Image(
                    label="End frame", visible=False, interactive=False,
                    show_download_button=False,
                    height=120, width=160, scale=0,
                )
            if HAS_RANGESLIDER:
                trim_range = RangeSlider(
                    0, 1, value=(0, 1), step=0.1,
                    label="Trim range (s)", show_label=False,
                )
            else:
                with gr.Row():
                    trim_start = gr.Slider(0, 1, value=0, step=0.1, label="Start (s)")
                    trim_end   = gr.Slider(0, 1, value=1, step=0.1, label="End (s)")

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

    if HAS_RANGESLIDER:
        files_to_process.upload(
            fn=on_file_upload_range,
            inputs=[files_to_process],
            outputs=[
                preview_image, preview_video,
                current_res_label, resize_resolution,
                current_fps_label, fps_value,
                fps_group,
                current_duration_label,
                trim_start_preview, trim_end_preview,
                trim_range,
                trim_group,
                file_info,
            ],
            show_progress="hidden",
        )

        trim_range.release(
            fn=on_trim_release_range,
            inputs=[files_to_process, trim_range],
            outputs=[trim_start_preview, trim_end_preview],
            show_progress="hidden",
        )

        btn_apply.click(
            fn=on_apply_all_range,
            inputs=[
                files_to_process,
                resize_resolution, rotation_choice,
                fps_value,
                trim_range,
                crop_left, crop_right, crop_top, crop_bottom,
                file_info,
            ],
            outputs=[output_image, output_video, output_path_state],
        )
    else:
        files_to_process.upload(
            fn=on_file_upload_pair,
            inputs=[files_to_process],
            outputs=[
                preview_image, preview_video,
                current_res_label, resize_resolution,
                current_fps_label, fps_value,
                fps_group,
                current_duration_label,
                trim_start_preview, trim_end_preview,
                trim_start, trim_end,
                trim_group,
                file_info,
            ],
            show_progress="hidden",
        )

        trim_start.release(
            fn=on_trim_release_pair,
            inputs=[files_to_process, trim_start, trim_end],
            outputs=[trim_start_preview, trim_end_preview],
            show_progress="hidden",
        )
        trim_end.release(
            fn=on_trim_release_pair,
            inputs=[files_to_process, trim_start, trim_end],
            outputs=[trim_start_preview, trim_end_preview],
            show_progress="hidden",
        )

        btn_apply.click(
            fn=on_apply_all_pair,
            inputs=[
                files_to_process,
                resize_resolution, rotation_choice,
                fps_value,
                trim_start, trim_end,
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


def _get_path(files):
    if not files:
        return None
    return files[0].name if hasattr(files[0], 'name') else str(files[0])


def _probe_file(files):
    """Shared detection logic for on_file_upload_range/on_file_upload_pair.

    Returns a dict of computed fields; {'valid': False} if there's no usable
    file (nothing uploaded, or an unsupported type).
    """
    path = _get_path(files)
    if path is None:
        return {"valid": False}

    is_awebp   = util.is_animated_webp(path)
    is_agif    = util.is_animated_gif(path)
    is_animated = is_awebp or is_agif
    is_img = util.is_image(path)   # returns False for animated webp/gif
    is_vid = util.is_video(path) or is_animated

    if not is_img and not is_vid:
        return {"valid": False}

    w, h = util.detect_dimensions(path)
    fps = util.detect_fps(path) if (is_vid or is_animated) else 24.0

    # Trim (and its frame previews) only applies to real, non-animated video.
    is_trimmable = is_vid and not is_animated
    duration = util.detect_duration(path) if is_trimmable else 0.0

    start_thumb = end_thumb = None
    if is_trimmable and duration > 0:
        start_thumb = util.extract_frame_at(path, 0.0)
        end_thumb   = util.extract_frame_at(path, duration)

    current_res = f"{w}x{h}" if w and h else RESOLUTION_CHOICES[0]
    choices = [current_res] + [r for r in RESOLUTION_CHOICES if r != current_res]

    info = {
        "width": w, "height": h, "fps": fps, "duration": duration,
        "is_video": is_vid, "is_animated_gif": is_agif, "is_animated_webp": is_awebp,
    }
    return {
        "valid": True, "path": path,
        "is_img": is_img, "is_vid": is_vid, "is_animated": is_animated,
        "w": w, "h": h, "fps": fps,
        "is_trimmable": is_trimmable, "duration": duration,
        "start_thumb": start_thumb, "end_thumb": end_thumb,
        "current_res": current_res, "choices": choices,
        "info": info,
    }


def _common_upload_updates(r):
    """Everything the two on_file_upload_* variants share — all outputs
    except the trim control itself (1 component for RangeSlider, 2 for the
    fallback pair), returned as (before, after) to be spliced around it."""
    if not r["valid"]:
        before = (
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(value="**Current:** —"),
            gr.update(choices=RESOLUTION_CHOICES, value=RESOLUTION_CHOICES[0]),
            gr.update(value="**Current:** —"),
            gr.update(value=30),
            gr.update(visible=False),
            gr.update(value="**Current:** —"),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )
        after = (gr.update(visible=False), dict(_EMPTY_INFO))
        return before, after

    show_as_img = r["is_img"] or r["is_animated"]
    show_as_vid = r["is_vid"] and not r["is_animated"]
    before = (
        gr.update(visible=show_as_img, value=r["path"] if show_as_img else None),
        gr.update(visible=show_as_vid, value=r["path"] if show_as_vid else None),
        gr.update(value=f"**Current:** {r['w']} × {r['h']}"),
        gr.update(choices=r["choices"], value=r["current_res"]),
        gr.update(value=f"**Current:** {r['fps']:.2f} fps"),
        gr.update(value=round(r["fps"])),
        gr.update(visible=r["is_vid"]),
        gr.update(value=f"**Current:** {r['duration']:.2f}s" if r["is_trimmable"] else "**Current:** —"),
        gr.update(visible=r["is_trimmable"], value=r["start_thumb"]),
        gr.update(visible=r["is_trimmable"], value=r["end_thumb"]),
    )
    after = (gr.update(visible=r["is_trimmable"]), r["info"])
    return before, after


def on_file_upload_range(files):
    r = _probe_file(files)
    before, after = _common_upload_updates(r)
    if not r["valid"]:
        trim_update = gr.update(minimum=0, maximum=1, value=(0, 1))
    else:
        dur = max(r["duration"], 0.1)
        trim_update = gr.update(minimum=0, maximum=dur, value=(0, r["duration"]), step=0.1)
    return before + (trim_update,) + after


def on_file_upload_pair(files):
    r = _probe_file(files)
    before, after = _common_upload_updates(r)
    if not r["valid"]:
        trim_updates = (
            gr.update(minimum=0, maximum=1, value=0),
            gr.update(minimum=0, maximum=1, value=1),
        )
    else:
        dur = max(r["duration"], 0.1)
        trim_updates = (
            gr.update(minimum=0, maximum=dur, value=0, step=0.1),
            gr.update(minimum=0, maximum=dur, value=r["duration"], step=0.1),
        )
    return before + trim_updates + after


def on_trim_release_range(files, trim_range):
    """Refresh the start/end thumbnails once the user lets go of a handle.

    Deliberately wired to `.release` rather than `.input` — reopening the
    video and seeking on every intermediate drag position would make
    dragging feel laggy; a single seek on release stays snappy.
    """
    path = _get_path(files)
    if path is None or trim_range is None:
        return gr.update(), gr.update()
    lo, hi = trim_range
    return (
        gr.update(value=util.extract_frame_at(path, lo)),
        gr.update(value=util.extract_frame_at(path, hi)),
    )


def on_trim_release_pair(files, trim_start, trim_end):
    path = _get_path(files)
    if path is None:
        return gr.update(), gr.update()
    return (
        gr.update(value=util.extract_frame_at(path, trim_start)),
        gr.update(value=util.extract_frame_at(path, trim_end)),
    )


def _apply_all_core(files, resolution, rotation, fps, trim_start, trim_end,
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
    cur_duration = file_info.get("duration", 0.0)
    print(f"[Editor] paths={paths}, is_vid={is_vid}, is_agif={is_agif}, cur_w={cur_w}, cur_h={cur_h}, cur_fps={cur_fps}, cur_duration={cur_duration}")

    # Trim only applies to real (non-animated) video — active whenever the
    # requested window is narrower than the detected duration.
    is_trimmable = is_vid and not is_agif and not is_awebp
    trim_active = (
        is_trimmable
        and trim_end is not None
        and trim_start is not None
        and trim_end > trim_start
        and (trim_start > 0.01 or trim_end < cur_duration - 0.01)
    )

    # Build vf filter list (order: crop → rotate → scale → fps)
    # Trim itself is applied separately (via -ss/-t, not -vf) — see below.
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

    print(f"[Editor] filters built: {filters}, trim_active={trim_active} ({trim_start}s → {trim_end}s)")
    if not filters and not trim_active and not (is_agif and abs(fps - cur_fps) > 0.1):
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
            success = ffmpeg.apply_media_transforms(
                f, dest, filters, is_vid,
                trim_start=trim_start if trim_active else 0.0,
                trim_end=trim_end if trim_active else None,
            )
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


def on_apply_all_range(files, resolution, rotation, fps, trim_range,
                       crop_left, crop_right, crop_top, crop_bottom,
                       file_info):
    trim_start, trim_end = trim_range if trim_range is not None else (0.0, 0.0)
    return _apply_all_core(
        files, resolution, rotation, fps, trim_start, trim_end,
        crop_left, crop_right, crop_top, crop_bottom, file_info,
    )


def on_apply_all_pair(files, resolution, rotation, fps, trim_start, trim_end,
                      crop_left, crop_right, crop_top, crop_bottom,
                      file_info):
    return _apply_all_core(
        files, resolution, rotation, fps, trim_start, trim_end,
        crop_left, crop_right, crop_top, crop_bottom, file_info,
    )


def on_send_to_faceswap(paths):
    if not paths:
        return None
    return paths
