import os
import shutil
import numpy as np
import gradio as gr
import roop.utilities as util
import roop.globals
import ui.globals
from roop.face_util import extract_face_images
from roop.capturer import get_video_frame, get_video_frame_total, get_image_frame
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions
from roop.FaceSet import FaceSet

last_image = None


SELECTED_INPUT_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0

input_faces = None
target_faces = None
previewimage = None

selected_preview_index = 0

is_processing = False            

list_files_process : list[ProcessEntry] = []
no_face_choices = ["Use untouched original frame","Retry rotated", "Skip Frame", "Skip Frame if no similar face", "Use last swapped"]
swap_choices = ["First found", "All input faces", "All female", "All male", "All faces", "Selected face"]

current_video_fps = 50

# Last swapped preview frame (numpy BGR), updated by on_preview_frame_changed.
# get_face_crop_for_mask reads this directly so previewimage never needs to be
# passed as a Gradio event input (which caused slider-value caching bugs).
_last_swapped_preview = None

# Last ProcessOptions used for a fake-preview swap.  Stored so _fbf_fetch_crop
# can re-run the swap for any FBF frame without needing all the Gradio inputs.
_last_preview_options = None


def faceswap_tab():
    global no_face_choices, previewimage

    with gr.Tab("🎭 Face Swap"):
        with gr.Row(variant='panel'):
            bt_srcfiles = gr.Files(label='Source Images or Facesets', file_count="multiple", file_types=["image", ".fsz"], elem_id='filelist', height=233)
            bt_destfiles = gr.Files(label='Target File(s)', file_count="multiple", file_types=["image", "video", ".webp"], elem_id='filelist', height=233)
        with gr.Row(variant='panel'):
            with gr.Column(scale=2):
                with gr.Row():
                    input_faces = gr.Gallery(label="Input faces gallery", allow_preview=False, preview=False, height=None, columns=2, object_fit="contain", interactive=False)
                    target_faces = gr.Gallery(label="Target faces gallery", allow_preview=False, preview=False, height=None, columns=2, object_fit="contain", interactive=False)
                with gr.Row():
                    bt_move_left_input = gr.Button("⬅ Move left", size='sm')
                    bt_move_right_input = gr.Button("➡ Move right", size='sm')
                    bt_move_left_target = gr.Button("⬅ Move left", size='sm')
                    bt_move_right_target = gr.Button("➡ Move right", size='sm')
                with gr.Row():
                    bt_remove_selected_input_face = gr.Button("❌ Remove selected", size='sm')
                    bt_clear_input_faces = gr.Button("💥 Clear all", variant='stop', size='sm')
                    bt_remove_selected_target_face = gr.Button("❌ Remove selected", size='sm')

                with gr.Row():
                    with gr.Column():
                        chk_showmaskoffsets = gr.Checkbox(
                            label="Show mask overlay in preview",
                            value=roop.globals.CFG.show_mask_offsets,
                            interactive=True,
                        )
                        chk_restoreoriginalmouth = gr.Checkbox(
                            label="Restore original mouth area",
                            value=roop.globals.CFG.restore_original_mouth,
                            interactive=True,
                        )
                        chk_use_3d_recon = gr.Checkbox(
                            label="🧊 3D source pose matching (experimental)",
                            value=roop.globals.CFG.use_3d_recon,
                            interactive=True,
                            info="Warps the source face to approximate the target head pose before embedding — improves profile and angled swaps.",
                        )
                        chk_use_source_bank = gr.Checkbox(
                            label="🎯 Multi-angle source bank",
                            value=roop.globals.CFG.use_source_bank,
                            interactive=True,
                            info="When a faceset has multiple source images, auto-selects the one whose pose best matches each target frame. Load a .fsz file with front + profile photos to use this.",
                        )
                        chk_use_frontalization = gr.Checkbox(
                            label="🏛 Frontalize before swap",
                            value=False,
                            interactive=False,
                            visible=False,
                        )
                        sld_frontalization_threshold = gr.Slider(
                            0.0, 90.0,
                            value=30.0,
                            step=1.0,
                            label="Frontalization threshold (°)",
                            interactive=False,
                            visible=False,
                        )
                        dd_swap_model = gr.Dropdown(
                            choices=["inswapper"],
                            value="inswapper",
                            label="🔀 Swap model",
                            interactive=False,
                            visible=False,
                        )
                        mask_top = gr.Slider(
                            0, 2.0, value=roop.globals.CFG.mask_top,
                            label="Offset Face Top", step=0.01, interactive=True,
                        )
                        mask_bottom = gr.Slider(
                            0, 2.0, value=roop.globals.CFG.mask_bottom,
                            label="Offset Face Bottom", step=0.01, interactive=True,
                        )
                        mask_left = gr.Slider(
                            0, 2.0, value=roop.globals.CFG.mask_left,
                            label="Offset Face Left", step=0.01, interactive=True,
                        )
                        mask_right = gr.Slider(
                            0, 2.0, value=roop.globals.CFG.mask_right,
                            label="Offset Face Right", step=0.01, interactive=True,
                        )
                        face_mask_blend = gr.Slider(
                            0, 200, value=roop.globals.CFG.face_mask_blend,
                            label="Face Mask Edge Blend", step=1, interactive=True,
                        )
                    with gr.Column():
                        mouth_top_scale = gr.Slider(
                            0, 2.0, value=roop.globals.CFG.mouth_top_scale,
                            label="Mouth Mask Top", step=0.01, interactive=True,
                        )
                        mouth_bottom_scale = gr.Slider(
                            0, 2.0, value=roop.globals.CFG.mouth_bottom_scale,
                            label="Mouth Mask Bottom", step=0.01, interactive=True,
                        )
                        mouth_left_scale = gr.Slider(
                            0, 2.0, value=roop.globals.CFG.mouth_left_scale,
                            label="Mouth Mask Left", step=0.01, interactive=True,
                        )
                        mouth_right_scale = gr.Slider(
                            0, 2.0, value=roop.globals.CFG.mouth_right_scale,
                            label="Mouth Mask Right", step=0.01, interactive=True,
                        )
                        mouth_mask_blend = gr.Slider(
                            0, 200, value=roop.globals.CFG.mouth_mask_blend,
                            label="Mouth Mask Edge Blend", step=1, interactive=True,
                        )
                        bt_toggle_masking = gr.Button(
                            "🎭 Edit Mask",
                            variant="primary",
                            elem_id="btn_toggle_masking",
                        )
                        selected_mask_engine = gr.Dropdown(
                            ["None", "Clip2Seg", "DFL XSeg"],
                            value=roop.globals.CFG.mask_engine,
                            label="Face masking engine",
                        )
                        clip_text = gr.Textbox(
                            label="List of objects to mask and restore back on fake face",
                            value=roop.globals.CFG.mask_clip_text,
                            interactive=roop.globals.CFG.mask_engine == "Clip2Seg",
                        )

            with gr.Column(scale=2):
                previewimage = gr.Image(label="Preview Image", height=576, interactive=False, visible=True, format=get_gradio_output_format(), elem_id="roop_preview_image")
                # mask_json_store: hidden textbox that holds the serialised dual-mask JSON written by the JS modal
                # visible="hidden" keeps the textarea in the DOM (tracked by Gradio) but takes no visual space.
                # visible=False would remove it from the DOM entirely (Svelte {#if} block), making it
                # unfindable by JS and excluded from Gradio's input payload — which was our bug.
                mask_json_store = gr.Textbox(value="", visible="hidden", elem_id="mask_json_store", label="Mask Data")
                # mask_per_frame_store: holds per-frame canvas masks.
                # New format: {"frame": {"facesetIdx": maskData, ...}, ...}
                # Old format: {"frame": maskData, ...}  (auto-upgraded on parse)
                mask_per_frame_store = gr.Textbox(value="", visible="hidden",
                                                   elem_id="mask_per_frame_store",
                                                   label="Per-Frame Mask Data")
                # mask_faceset_count_store: number of loaded source facesets.
                # Updated whenever INPUT_FACESETS changes so the mask editor can build
                # its faceset selector strip without a Python round-trip.
                mask_faceset_count_store = gr.Textbox(value="0", visible="hidden",
                                                       elem_id="mask_faceset_count_store",
                                                       label="Faceset Count")
                # fbf_frame_num_store: JS writes "frameNum:seq" here from _fbfFetchFaceCrop.
                # The .change() event on this textbox triggers the Python face-crop fetch.
                # (A hidden button was tried but gr.Button(visible=False) is not rendered
                # to the DOM in Gradio 5, making programmatic clicks impossible.)
                fbf_frame_num_store = gr.Textbox(value="1", visible="hidden",
                                                  elem_id="fbf_frame_num_store",
                                                  label="FBF Frame Number")
                # mask_kps_store: holds the 5-point face keypoints (JSON) of the reference frame
                # where the mask was painted; embedded in the mask JSON for per-frame tracking.
                mask_kps_store = gr.Textbox(value="", visible="hidden", elem_id="mask_kps_store", label="Mask KPS")
                # mask_face_crop_store: holds the canonical 512×512 face crop as a
                # base64 PNG data-URL.  The mask editor reads this and uses it as the
                # drawing-surface background, so the mask is painted in face-crop
                # coordinate space and always tracks the face at processing time.
                # A hidden Textbox (not gr.Image) guarantees the value is in the DOM
                # immediately when the follow-on JS runs.
                mask_face_crop_store = gr.Textbox(value="", visible="hidden",
                                                  elem_id="mask_face_crop_store",
                                                  label="Face Crop Data URL")
                # mask_face_swap_crop_store: the swapped (post-swap) face crop as a
                # base64 data-URL, used as the live-preview base in the mask editor.
                # Empty when no swap preview has been generated yet.
                mask_face_swap_crop_store = gr.Textbox(value="", visible="hidden",
                                                       elem_id="mask_face_swap_crop_store",
                                                       label="Swapped Face Crop Data URL")
                # original_frame_img: stores the unswapped source frame so the masking editor
                # always shows the original image, not the face-swapped result.
                # visible="hidden" keeps it in the DOM (needed by JS) but takes no visual space.
                original_frame_img = gr.Image(value=None, visible="hidden", elem_id="roop_original_frame",
                                              label="Original Frame", format=get_gradio_output_format(), interactive=False)
                with gr.Row(variant='panel'):
                    fake_preview = gr.Checkbox(label="Face swap frames", value=False)
                    bt_refresh_preview = gr.Button("🔄 Refresh", variant='secondary', size='sm', elem_id="btn_refresh_preview")
                    bt_use_face_from_preview = gr.Button("Use Face from this Frame", variant='primary', size='sm')
                with gr.Row():
                    preview_frame_num = gr.Slider(1, 1, value=1, label="Frame Number", info='0:00:00', step=1.0, interactive=True,
                                                   elem_id="preview_frame_num")
                with gr.Row():
                    text_frame_clip = gr.Markdown('Processing frame range [0 - 0]')
                    set_frame_start = gr.Button("⬅ Set as Start", size='sm')
                    set_frame_end = gr.Button("➡ Set as End", size='sm')
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                selected_face_detection = gr.Dropdown(swap_choices, value=roop.globals.CFG.face_detection_mode, label="Specify face selection for swapping")
            with gr.Column(scale=1):
                num_swap_steps = gr.Slider(1, 5, value=roop.globals.CFG.num_swap_steps, step=1.0, label="Number of swapping steps", info="More steps may increase likeness")
            with gr.Column(scale=2):
                ui.globals.ui_selected_enhancer = gr.Dropdown(["None", "Codeformer", "DMDNet", "GFPGAN", "GPEN", "Restoreformer++"], value=roop.globals.CFG.selected_enhancer, label="Select post-processing")

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                max_face_distance = gr.Slider(0.01, 1.0, value=roop.globals.CFG.max_face_distance, label="Max Face Similarity Threshold", info="0.0 = identical 1.0 = no similarity", elem_id='max_face_distance', interactive=True)
            with gr.Column(scale=1):
                ui.globals.ui_upscale = gr.Dropdown(["128px", "256px", "512px"], value=roop.globals.CFG.subsample_upscale, label="Subsample upscale to", interactive=True)
            with gr.Column(scale=2):
                ui.globals.ui_blend_ratio = gr.Slider(0.0, 1.0, value=roop.globals.CFG.blend_ratio, label="Original/Enhanced image blend ratio", info="Only used with active post-processing")

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                video_swapping_method = gr.Dropdown(["Extract Frames to media","In-Memory processing"], value=roop.globals.CFG.video_swapping_method, label="Select video processing method", interactive=True)
                no_face_action = gr.Dropdown(choices=no_face_choices, value=roop.globals.CFG.no_face_action, label="Action on no face detected", interactive=True)
                vr_mode = gr.Checkbox(label="VR Mode", value=roop.globals.CFG.vr_mode)
            with gr.Column(scale=1):
                with gr.Group():
                    autorotate = gr.Checkbox(label="Auto rotate horizontal Faces", value=roop.globals.CFG.autorotate_faces)
                    roop.globals.skip_audio = gr.Checkbox(label="Skip audio", value=roop.globals.CFG.skip_audio)
                    roop.globals.keep_frames = gr.Checkbox(label="Keep Frames (relevant only when extracting frames)", value=roop.globals.CFG.keep_frames)
                    roop.globals.wait_after_extraction = gr.Checkbox(label="Wait for user key press before creating video ", value=roop.globals.CFG.wait_after_extraction)

        with gr.Row(variant='panel'):
            with gr.Column():
                bt_start = gr.Button("▶ Start", variant='primary')
            with gr.Column():
                bt_stop = gr.Button("⏹ Stop", variant='secondary', interactive=False)
                gr.Button("👀 Open Output Folder", size='sm').click(fn=lambda: util.open_folder(roop.globals.output_path))
            with gr.Column(scale=2):
                output_method = gr.Dropdown(["File","Virtual Camera", "Both"], value=roop.globals.CFG.output_method, label="Select Output Method", interactive=True)

        # No gr.HTML modal component needed — the masking modal is created entirely by
        # JavaScript injected via Blocks(head=MASKING_HEAD_JS) in main.py.
        # Gradio 5 strips <script> tags from gr.HTML, so all JS must go through head=.

    # Store saveable component refs in ui.globals for cross-tab access (Save/Load session)
    ui.globals.ui_selected_face_detection = selected_face_detection
    ui.globals.ui_num_swap_steps = num_swap_steps
    ui.globals.ui_max_face_distance = max_face_distance
    ui.globals.ui_video_swapping_method = video_swapping_method
    ui.globals.ui_no_face_action = no_face_action
    ui.globals.ui_vr_mode = vr_mode
    ui.globals.ui_autorotate = autorotate
    ui.globals.ui_skip_audio = roop.globals.skip_audio
    ui.globals.ui_keep_frames = roop.globals.keep_frames
    ui.globals.ui_wait_after_extraction = roop.globals.wait_after_extraction
    ui.globals.ui_output_method = output_method
    ui.globals.ui_selected_mask_engine = selected_mask_engine
    ui.globals.ui_clip_text = clip_text
    ui.globals.ui_chk_showmaskoffsets = chk_showmaskoffsets
    ui.globals.ui_chk_restoreoriginalmouth = chk_restoreoriginalmouth
    ui.globals.ui_chk_use_3d_recon = chk_use_3d_recon
    ui.globals.ui_chk_use_source_bank = chk_use_source_bank
    ui.globals.ui_chk_use_frontalization = chk_use_frontalization
    ui.globals.ui_sld_frontalization_threshold = sld_frontalization_threshold
    ui.globals.ui_dd_swap_model = dd_swap_model
    ui.globals.ui_mask_top = mask_top
    ui.globals.ui_mask_bottom = mask_bottom
    ui.globals.ui_mask_left = mask_left
    ui.globals.ui_mask_right = mask_right
    ui.globals.ui_face_mask_blend = face_mask_blend
    ui.globals.ui_mouth_mask_blend = mouth_mask_blend
    ui.globals.ui_mouth_top_scale = mouth_top_scale
    ui.globals.ui_mouth_bottom_scale = mouth_bottom_scale
    ui.globals.ui_mouth_left_scale = mouth_left_scale
    ui.globals.ui_mouth_right_scale = mouth_right_scale

    previewinputs = [preview_frame_num, bt_destfiles, fake_preview, ui.globals.ui_selected_enhancer, selected_face_detection,
                        max_face_distance, ui.globals.ui_blend_ratio, selected_mask_engine, clip_text, no_face_action, vr_mode, autorotate, mask_json_store, chk_showmaskoffsets, chk_restoreoriginalmouth, num_swap_steps, ui.globals.ui_upscale,
                        chk_use_3d_recon, mask_per_frame_store,
                        chk_use_source_bank, chk_use_frontalization, sld_frontalization_threshold, dd_swap_model]
    previewoutputs = [previewimage, preview_frame_num, original_frame_img]
    input_faces.select(on_select_input_face, None, None).success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)
    
    bt_move_left_input.click(fn=move_selected_input, inputs=[bt_move_left_input], outputs=[input_faces])
    bt_move_right_input.click(fn=move_selected_input, inputs=[bt_move_right_input], outputs=[input_faces])
    bt_move_left_target.click(fn=move_selected_target, inputs=[bt_move_left_target], outputs=[target_faces])
    bt_move_right_target.click(fn=move_selected_target, inputs=[bt_move_right_target], outputs=[target_faces])

    bt_remove_selected_input_face.click(fn=remove_selected_input_face, outputs=[input_faces, mask_faceset_count_store])
    bt_srcfiles.upload(fn=on_srcfile_changed, show_progress='full', inputs=bt_srcfiles, outputs=[input_faces, bt_srcfiles, mask_faceset_count_store])

    mask_top.release(fn=on_mask_top_changed, inputs=[mask_top], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mask_bottom.release(fn=on_mask_bottom_changed, inputs=[mask_bottom], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mask_left.release(fn=on_mask_left_changed, inputs=[mask_left], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mask_right.release(fn=on_mask_right_changed, inputs=[mask_right], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    face_mask_blend.release(fn=on_face_mask_blend_changed, inputs=[face_mask_blend], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mouth_mask_blend.release(fn=on_mouth_mask_blend_changed, inputs=[mouth_mask_blend], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mouth_top_scale.release(fn=on_mouth_top_scale_changed, inputs=[mouth_top_scale], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mouth_bottom_scale.release(fn=on_mouth_bottom_scale_changed, inputs=[mouth_bottom_scale], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mouth_left_scale.release(fn=on_mouth_left_scale_changed, inputs=[mouth_left_scale], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    mouth_right_scale.release(fn=on_mouth_right_scale_changed, inputs=[mouth_right_scale], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    chk_showmaskoffsets.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    chk_restoreoriginalmouth.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    selected_mask_engine.change(fn=on_mask_engine_changed, inputs=[selected_mask_engine], outputs=[clip_text], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')

    target_faces.select(on_select_target_face, None, None)
    bt_remove_selected_target_face.click(fn=remove_selected_target_face, outputs=[target_faces])

    bt_destfiles.change(fn=on_destfiles_changed, inputs=[bt_destfiles], outputs=[preview_frame_num, text_frame_clip], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    bt_destfiles.select(fn=on_destfiles_selected, outputs=[preview_frame_num, text_frame_clip], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    bt_destfiles.clear(fn=on_clear_destfiles, outputs=[target_faces, mask_json_store, preview_frame_num, text_frame_clip]).then(
        fn=None, js="() => { if (window.maskReset) maskReset(); }"
    )
    bt_clear_input_faces.click(fn=on_clear_input_faces, outputs=[input_faces, mask_faceset_count_store])


    start_event = bt_start.click(fn=start_swap,
        inputs=[output_method, ui.globals.ui_selected_enhancer, selected_face_detection, roop.globals.keep_frames, roop.globals.wait_after_extraction,
                    roop.globals.skip_audio, max_face_distance, ui.globals.ui_blend_ratio, selected_mask_engine, clip_text, video_swapping_method, no_face_action, vr_mode, autorotate, chk_restoreoriginalmouth, num_swap_steps, ui.globals.ui_upscale, mask_json_store,
                    chk_use_3d_recon, mask_per_frame_store,
                    chk_use_source_bank, chk_use_frontalization, sld_frontalization_threshold, dd_swap_model],
        outputs=[bt_start, bt_stop], show_progress='full')

    bt_stop.click(fn=stop_swap, cancels=[start_event], outputs=[bt_start, bt_stop], queue=False)

    bt_refresh_preview.click(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)
    # Pure client-side toggle — no Python round-trip needed.
    # maskToggle() is defined in MASKING_HEAD_JS injected via Blocks(head=) in main.py.
    bt_toggle_masking.click(
        fn=get_face_crop_for_mask,
        inputs=[preview_frame_num, bt_destfiles],
        outputs=[mask_face_crop_store, mask_face_swap_crop_store],
        show_progress='hidden'
    ).then(fn=None, js="() => maskToggle()")
    # FBF in-modal frame navigation: JS writes "frameNum:seq" to fbf_frame_num_store.
    # gr.Files components are not reliably serialized in .change() events in Gradio 5
    # (they arrive as None), so we bypass bt_destfiles entirely and use
    # roop.globals.target_path which is already set when the user loaded their target.
    def _fbf_fetch_crop(frame_num_str):
        global _last_swapped_preview
        # frame_num_str format: "frameNum:facesetIdx:seq"
        # (Legacy format "N:seq" also handled — faceset defaults to 0.)
        parts = (frame_num_str or "1:0:1").split(":")
        part = parts[0]
        frame_num = int(float(part)) if part else 1
        faceset_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        # Use list_files_process / selected_preview_index — same globals the
        # preview system uses; roop.globals.target_path is only set by
        # on_use_face_from_selected and is None during normal preview use.
        if not list_files_process or selected_preview_index >= len(list_files_process):
            print(f"[FBF] no files loaded (list_files_process={list_files_process!r})")
            return "", ""
        filename = list_files_process[selected_preview_index].filename
        print(f"[FBF] frame_num={frame_num}  filename={filename!r}")

        # Shim: get_face_crop_for_mask expects files[idx].name
        class _F:
            def __init__(self, p):
                self.name = p

        # When fake_preview was active, regenerate the swap for this specific
        # FBF frame using the stored ProcessOptions.  We temporarily patch
        # _last_swapped_preview so get_face_crop_for_mask picks up the correct
        # swap; the original is always restored afterward.
        # util == roop.utilities (module-level alias)
        saved_swap = _last_swapped_preview
        if _last_preview_options is not None:
            try:
                from roop.core import live_swap as _live_swap
                if util.is_video(filename) or filename.lower().endswith('gif') or util.is_animated_webp(filename):
                    raw_frame = get_video_frame(filename, frame_num)
                else:
                    raw_frame = get_image_frame(filename)
                if raw_frame is not None:
                    swapped = _live_swap(raw_frame.copy(), _last_preview_options)
                    _last_swapped_preview = swapped  # may be None on failure
                    print(f"[FBF] live_swap for frame {frame_num}: {'ok' if swapped is not None else 'failed'}")
            except Exception as _e:
                print(f"[FBF] swap preview failed: {_e}")

        result = get_face_crop_for_mask(frame_num, [_F(filename)], faceset_idx)

        # Restore so the main-preview cache is unchanged.
        _last_swapped_preview = saved_swap

        print(f"[FBF] result: src_len={len(result[0])}  swp_len={len(result[1])}")
        return result
    fbf_frame_num_store.change(
        fn=_fbf_fetch_crop,
        inputs=[fbf_frame_num_store],
        outputs=[mask_face_crop_store, mask_face_swap_crop_store],
        show_progress='hidden',
    ).then(fn=None, js="""() => {
      console.log('[FBF] .then fired — reading from DOM stores');
      if (window._fbfOnCropReady) window._fbfOnCropReady();
      else console.warn('[FBF] _fbfOnCropReady is null');
    }""")
    fake_preview.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)
    preview_frame_num.release(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')

    # ── Settings that were previously missing auto-refresh ──────────────────
    ui.globals.ui_selected_enhancer.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    selected_face_detection.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    max_face_distance.release(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    ui.globals.ui_blend_ratio.release(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    clip_text.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    no_face_action.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    vr_mode.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    autorotate.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    num_swap_steps.release(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    ui.globals.ui_upscale.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    chk_use_3d_recon.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    chk_use_source_bank.change(fn=on_use_source_bank_changed, inputs=[chk_use_source_bank], show_progress='hidden').success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden')
    # chk_use_frontalization, sld_frontalization_threshold, dd_swap_model are hidden/disabled

    bt_use_face_from_preview.click(fn=on_use_face_from_selected, show_progress='full', inputs=[bt_destfiles, preview_frame_num], outputs=[target_faces, selected_face_detection])
    set_frame_start.click(fn=on_set_frame, inputs=[set_frame_start, preview_frame_num], outputs=[text_frame_clip])
    set_frame_end.click(fn=on_set_frame, inputs=[set_frame_end, preview_frame_num], outputs=[text_frame_clip])

    return bt_destfiles


def on_mask_top_changed(mask_offset):
    set_mask_offset(0, mask_offset)

def on_mask_bottom_changed(mask_offset):
    set_mask_offset(1, mask_offset)

def on_mask_left_changed(mask_offset):
    set_mask_offset(2, mask_offset)

def on_mask_right_changed(mask_offset):
    set_mask_offset(3, mask_offset)

def on_face_mask_blend_changed(value):
    set_mask_offset(4, value)

def on_mouth_mask_blend_changed(value):
    set_mask_offset(5, value)

def on_mouth_top_scale_changed(value):
    set_mask_offset(6, value)

def on_mouth_bottom_scale_changed(value):
    set_mask_offset(7, value)

def on_mouth_left_scale_changed(value):
    set_mask_offset(8, value)

def on_mouth_right_scale_changed(value):
    set_mask_offset(9, value)

def set_mask_offset(index, mask_offset):
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        offs = roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets
        # Indices 6-9 are mouth scales with a default of 1.0 — pad correctly.
        while len(offs) < 10:
            offs.append(1.0)
        offs[index] = mask_offset
        roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = offs

def on_use_source_bank_changed(value):
    roop.globals.CFG.use_source_bank = value
    roop.globals.CFG.save()

def on_use_frontalization_changed(value):
    roop.globals.CFG.use_frontalization = value
    roop.globals.CFG.save()

def on_frontalization_threshold_changed(value):
    roop.globals.CFG.frontalization_threshold = value
    roop.globals.CFG.save()

def on_swap_model_changed(value):
    roop.globals.CFG.swap_model = value
    roop.globals.CFG.save()

def on_mask_engine_changed(mask_engine):
    if mask_engine == "Clip2Seg":
        return gr.Textbox(interactive=True)
    return gr.Textbox(interactive=False)



def on_srcfile_changed(srcfiles, progress=gr.Progress()):
    global input_faces, last_image

    if srcfiles is None or len(srcfiles) < 1:
        return ui.globals.ui_input_thumbs, None

    for f in srcfiles:    
        source_path = f.name
        if source_path.lower().endswith('fsz'):
            progress(0, desc="Retrieving faces from Faceset File")      
            unzipfolder = os.path.join(os.environ["TEMP"], 'faceset')
            if os.path.isdir(unzipfolder):
                files = os.listdir(unzipfolder)
                for file in files:
                    os.remove(os.path.join(unzipfolder, file))
            else:
                os.makedirs(unzipfolder)
            util.mkdir_with_umask(unzipfolder)
            util.unzip(source_path, unzipfolder)
            is_first = True
            face_set = FaceSet()
            for file in os.listdir(unzipfolder):
                if file.endswith(".png"):
                    filename = os.path.join(unzipfolder,file)
                    progress(0, desc="Extracting faceset")      
                    selection_faces_data = extract_face_images(filename,  (False, 0))
                    for f in selection_faces_data:
                        face = f[0]
                        face.mask_offsets = [
                            roop.globals.CFG.mask_top,
                            roop.globals.CFG.mask_bottom,
                            roop.globals.CFG.mask_left,
                            roop.globals.CFG.mask_right,
                            roop.globals.CFG.face_mask_blend,
                            roop.globals.CFG.mouth_mask_blend,
                            roop.globals.CFG.mouth_top_scale,
                            roop.globals.CFG.mouth_bottom_scale,
                            roop.globals.CFG.mouth_left_scale,
                            roop.globals.CFG.mouth_right_scale,
                        ]
                        face_set.faces.append(face)
                        if is_first:
                            image = util.convert_to_gradio(f[1])
                            ui.globals.ui_input_thumbs.append(image)
                            is_first = False
                        face_set.ref_images.append(get_image_frame(filename))
            if len(face_set.faces) > 0:
                if len(face_set.faces) > 1:
                    face_set.AverageEmbeddings()
                roop.globals.INPUT_FACESETS.append(face_set)
                                        
        elif util.has_image_extension(source_path):
            progress(0, desc="Retrieving faces from image")      
            roop.globals.source_path = source_path
            selection_faces_data = extract_face_images(roop.globals.source_path,  (False, 0))
            progress(0.5, desc="Retrieving faces from image")
            for f in selection_faces_data:
                face_set = FaceSet()
                face = f[0]
                face.mask_offsets = [
                    roop.globals.CFG.mask_top,
                    roop.globals.CFG.mask_bottom,
                    roop.globals.CFG.mask_left,
                    roop.globals.CFG.mask_right,
                    roop.globals.CFG.face_mask_blend,
                    roop.globals.CFG.mouth_mask_blend,
                    roop.globals.CFG.mouth_top_scale,
                    roop.globals.CFG.mouth_bottom_scale,
                    roop.globals.CFG.mouth_left_scale,
                    roop.globals.CFG.mouth_right_scale,
                ]
                face_set.faces.append(face)
                image = util.convert_to_gradio(f[1])
                ui.globals.ui_input_thumbs.append(image)
                roop.globals.INPUT_FACESETS.append(face_set)
                
    progress(1.0)
    if len(ui.globals.ui_input_thumbs) >= 6:
        gr.Warning(
            "You have more than 6 input faces. Consider using the Face Management tab "
            "to consolidate multiple images of the same source into a single faceset file."
        )
    return ui.globals.ui_input_thumbs, None, str(len(roop.globals.INPUT_FACESETS))


def on_select_input_face(evt: gr.SelectData):
    global SELECTED_INPUT_FACE_INDEX

    SELECTED_INPUT_FACE_INDEX = evt.index


def remove_selected_input_face():
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        f = roop.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
        del f
    if len(ui.globals.ui_input_thumbs) > SELECTED_INPUT_FACE_INDEX:
        f = ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
        del f

    return ui.globals.ui_input_thumbs, str(len(roop.globals.INPUT_FACESETS))

def move_selected_input(button_text):
    global SELECTED_INPUT_FACE_INDEX

    if button_text == "⬅ Move left":
        if SELECTED_INPUT_FACE_INDEX <= 0:
            return ui.globals.ui_input_thumbs
        offset = -1
    else:
        if len(ui.globals.ui_input_thumbs) <= SELECTED_INPUT_FACE_INDEX:
            return ui.globals.ui_input_thumbs
        offset = 1
    
    f = roop.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
    roop.globals.INPUT_FACESETS.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    f = ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
    ui.globals.ui_input_thumbs.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    return ui.globals.ui_input_thumbs
        

def move_selected_target(button_text):
    global SELECTED_TARGET_FACE_INDEX

    if button_text == "⬅ Move left":
        if SELECTED_TARGET_FACE_INDEX <= 0:
            return ui.globals.ui_target_thumbs
        offset = -1
    else:
        if len(ui.globals.ui_target_thumbs) <= SELECTED_TARGET_FACE_INDEX:
            return ui.globals.ui_target_thumbs
        offset = 1
    
    f = roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
    roop.globals.TARGET_FACES.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    f = ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
    ui.globals.ui_target_thumbs.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    return ui.globals.ui_target_thumbs




def on_select_target_face(evt: gr.SelectData):
    global SELECTED_TARGET_FACE_INDEX

    SELECTED_TARGET_FACE_INDEX = evt.index

def remove_selected_target_face():
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    return ui.globals.ui_target_thumbs


def on_use_face_from_selected(files, frame_num):
    roop.globals.target_path = files[selected_preview_index].name
    faces_data = []

    if util.is_image(roop.globals.target_path) and not roop.globals.target_path.lower().endswith(('gif')):
        faces_data = extract_face_images(roop.globals.target_path, (False, 0))
    elif util.is_video(roop.globals.target_path) or roop.globals.target_path.lower().endswith(('gif')) or util.is_animated_webp(roop.globals.target_path):
        faces_data = extract_face_images(roop.globals.target_path, (True, frame_num))
    else:
        gr.Info('Unknown image/video type!')
        roop.globals.target_path = None
        return ui.globals.ui_target_thumbs, gr.Dropdown(visible=True)

    if len(faces_data) == 0:
        gr.Info('No faces detected!')
        roop.globals.target_path = None
        return ui.globals.ui_target_thumbs, gr.Dropdown(visible=True)

    for f in faces_data:
        roop.globals.TARGET_FACES.append(f[0])
        ui.globals.ui_target_thumbs.append(util.convert_to_gradio(f[1]))

    return ui.globals.ui_target_thumbs, gr.Dropdown(value='Selected face')


def get_face_crop_for_mask(frame_num, files, faceset_index=None):
    """Return two base64 PNG data-URLs: (source_face_crop, swapped_face_crop).

    source_face_crop — 512×512 canonical face crop that exactly matches the
      coordinate space ProcessMgr operates in.  When autorotate_faces is active
      and the face is horizontal the function replicates the cutout+rotate step
      from process_face before running align_crop, so the editor background and
      the processor canonical space are identical regardless of head angle.

    swapped_face_crop — same canonical crop computed from the swap preview frame.
      Used as the live-preview base so the user sees the swap result with mask
      overlays.  Empty string when no swap preview is available.

    faceset_index — which source faceset to target.  When multiple faces are
      detected in the target frame the function finds the one whose embedding
      best matches INPUT_FACESETS[faceset_index].  Defaults to
      SELECTED_INPUT_FACE_INDEX when None.

    Both return "" on failure so the hidden Textboxes fall back gracefully."""
    import base64 as _b64
    import cv2 as _cv2
    import numpy as _np
    from roop.face_util import get_first_face, get_all_faces, align_crop, rotate_anticlockwise, rotate_clockwise
    import roop.globals

    if faceset_index is None:
        faceset_index = SELECTED_INPUT_FACE_INDEX

    def _rotation_action(face, frame):
        """Mirror ProcessMgr.rotation_action — returns direction string or None."""
        bbox_w = face.bbox[2] - face.bbox[0]
        bbox_h = face.bbox[3] - face.bbox[1]
        if bbox_w <= bbox_h:
            return None  # upright face — no rotation needed
        # Horizontal face: use chin/forehead landmarks to pick direction
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            forehead_x = face.landmark_2d_106[72][0]
            chin_x     = face.landmark_2d_106[0][0]
            if chin_x < forehead_x:
                return "rotate_anticlockwise"
            if forehead_x < chin_x:
                return "rotate_clockwise"
        # Landmark fallback: use bbox centre vs frame centre
        fh, fw = frame.shape[:2]
        bbox_cx = face.bbox[0] + bbox_w / 2.0
        return "rotate_anticlockwise" if bbox_cx >= fw / 2.0 else "rotate_clockwise"

    def _cutout(frame, x0, y0, x1, y1):
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(frame.shape[1], int(x1)); y1 = min(frame.shape[0], int(y1))
        return frame[y0:y1, x0:x1]

    def _best_face_for_faceset(frame, fs_idx):
        """Return the target face in *frame* that best matches source faceset fs_idx.

        Falls back to the first detected face when only one face is present,
        when the faceset has no embedding, or when face detection fails."""
        all_f = get_all_faces(frame)
        if not all_f:
            return None
        if len(all_f) == 1 or fs_idx >= len(roop.globals.INPUT_FACESETS):
            return all_f[0]
        src_faceset = roop.globals.INPUT_FACESETS[fs_idx]
        if not src_faceset.faces or not hasattr(src_faceset.faces[0], 'embedding'):
            return all_f[0]
        src_emb = src_faceset.faces[0].embedding
        best_face, best_dist = all_f[0], float('inf')
        for f in all_f:
            if hasattr(f, 'embedding'):
                d = util.compute_cosine_distance(src_emb, f.embedding)
                if d < best_dist:
                    best_dist = d
                    best_face = f
        return best_face

    def _get_aligned_crop_params(frame, target_face=None):
        """Detect or use *target_face* in *frame*, apply autorotation, return (aligned_frame, kps, swap_fn).

        Returns (None, None, None) if no usable face is found.

        swap_fn is a closure that applies the EXACT same spatial transform
        (cutout + rotation, or identity) to any other frame so its coordinate
        space matches aligned_frame.  This guarantees src_kps are valid for both
        the source and the swap crops without any independent recomputation."""
        if frame is None:
            return None, None, None
        face = target_face if target_face is not None else get_first_face(frame)
        if face is None or not hasattr(face, 'kps') or face.kps is None:
            return None, None, None
        if roop.globals.autorotate_faces:
            action = _rotation_action(face, frame)
            if action is not None:
                x0, y0, x1, y1 = face.bbox.astype(int)
                offs = int(max(x1 - x0, y1 - y0) * 0.25)
                cut = _cutout(frame, x0 - offs, y0 - offs, x1 + offs, y1 + offs)
                rot = rotate_anticlockwise(cut) if action == "rotate_anticlockwise" else rotate_clockwise(cut)
                rotface = get_first_face(rot)
                if rotface is not None and hasattr(rotface, 'kps') and rotface.kps is not None:
                    # Capture loop variables explicitly so the closure is correct.
                    _x0, _y0, _x1, _y1, _offs, _act = x0, y0, x1, y1, offs, action
                    def _swap_fn(swp, __x0=_x0, __y0=_y0, __x1=_x1, __y1=_y1,
                                 __offs=_offs, __act=_act):
                        c = _cutout(swp, __x0 - __offs, __y0 - __offs,
                                        __x1 + __offs, __y1 + __offs)
                        return rotate_anticlockwise(c) if __act == "rotate_anticlockwise" \
                               else rotate_clockwise(c)
                    return rot, rotface.kps, _swap_fn
        # No rotation — identity transform for the swap frame too.
        return frame, face.kps, lambda swp: swp

    def _crop_with_kps(frame, kps):
        """Return a base64 PNG data-URL for align_crop(frame, kps, 512)."""
        if frame is None or kps is None:
            return ""
        crop, _ = align_crop(frame, kps, 512)
        ok, buf = _cv2.imencode('.png', crop)
        if not ok:
            return ""
        return "data:image/png;base64," + _b64.b64encode(buf.tobytes()).decode('utf-8')

    # --- Source face crop ---
    if files is None or selected_preview_index >= len(files) or frame_num is None:
        return "", ""
    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif') or util.is_animated_webp(filename):
        current_frame = get_video_frame(filename, frame_num)
    else:
        current_frame = get_image_frame(filename)

    # Find the target face for this faceset, then detect ONCE.
    # swap_fn applies the identical spatial transform to the swap frame so
    # src_kps remain valid without independent re-detection.
    matched_face = _best_face_for_faceset(current_frame, faceset_index)
    src_aligned, src_kps, swap_fn = _get_aligned_crop_params(current_frame, matched_face)
    src_url = _crop_with_kps(src_aligned, src_kps)

    # --- Swapped face crop ---
    # CRITICAL: use the same keypoints from the source face to crop the swap frame.
    # Re-detecting the face on the swap frame gives slightly different kps because
    # the inserted face is not pixel-identical to the source, which shifts the
    # canonical crop and misaligns the painted mask.
    swp_url = ""
    if _last_swapped_preview is not None and src_kps is not None and swap_fn is not None:
        swap_aligned = swap_fn(_last_swapped_preview)
        swp_url = _crop_with_kps(swap_aligned, src_kps)

    return src_url, swp_url


def on_preview_frame_changed(frame_num, files, fake_preview, enhancer, detection, face_distance, blend_ratio,
                              selected_mask_engine, clip_text, no_face_action, vr_mode, auto_rotate, mask_json, show_face_area, restore_original_mouth, num_steps, upsample,
                              use_3d_recon=False, mask_per_frame_json="",
                              use_source_bank=False, use_frontalization=False,
                              frontalization_threshold=25.0, swap_model='inswapper'):
    global SELECTED_INPUT_FACE_INDEX, current_video_fps, _last_swapped_preview, _last_preview_options

    from roop.core import live_swap, get_processing_plugins

    # If there is a per-frame mask for this specific frame, use it instead of
    # (or in addition to) the global mask_json.  Per-frame masks take priority.
    if mask_per_frame_json:
        import json as _json
        try:
            _pfm = _json.loads(mask_per_frame_json)
            _key = str(int(frame_num)) if frame_num is not None else None
            if _key and _key in _pfm:
                _frame_data = _pfm[_key]
                if isinstance(_frame_data, dict):
                    # Old flat format: {exclude, canonical} — wrap for ProcessMgr new format
                    _is_old = any(x in _frame_data for x in ('exclude', 'include', 'canonical'))
                    if _is_old:
                        mask_json = _json.dumps({"0": _frame_data})
                    else:
                        # New format: {facesetIdx: maskData} — pass as-is (ProcessMgr handles it)
                        mask_json = _json.dumps(_frame_data)
        except Exception:
            pass

    mask_offsets = [
        roop.globals.CFG.mask_top,
        roop.globals.CFG.mask_bottom,
        roop.globals.CFG.mask_left,
        roop.globals.CFG.mask_right,
        roop.globals.CFG.face_mask_blend,
        roop.globals.CFG.mouth_mask_blend,
        roop.globals.CFG.mouth_top_scale,
        roop.globals.CFG.mouth_bottom_scale,
        roop.globals.CFG.mouth_left_scale,
        roop.globals.CFG.mouth_right_scale,
    ]
    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        if not hasattr(roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0], 'mask_offsets'):
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = list(mask_offsets)
        mask_offsets = roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets
        while len(mask_offsets) < 10:
            mask_offsets.append(1.0)   # indices 6-9 are mouth scales, default 1.0

    timeinfo = '0:00:00'
    if files is None or selected_preview_index >= len(files) or frame_num is None:
        return None, gr.Slider(info=timeinfo), None

    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif') or util.is_animated_webp(filename):
        current_frame = get_video_frame(filename, frame_num)
        if current_video_fps == 0:
            current_video_fps = 1
        secs = (frame_num - 1) / current_video_fps
        minutes = secs / 60
        secs = secs % 60
        hours = minutes / 60
        minutes = minutes % 60
        milliseconds = (secs - int(secs)) * 1000
        timeinfo = f"{int(hours):0>2}:{int(minutes):0>2}:{int(secs):0>2}.{int(milliseconds):0>3}"
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return None, gr.Slider(info=timeinfo), None

    # Capture the original frame (before any face swap) for the masking editor.
    # convert_to_gradio returns a new RGB array so original_frame is not mutated by live_swap.
    original_frame = util.convert_to_gradio(current_frame)

    if not fake_preview or len(roop.globals.INPUT_FACESETS) < 1:
        _last_swapped_preview = None
        _last_preview_options = None
        return (gr.Image(value=original_frame, visible=True),
                gr.Slider(info=timeinfo),
                gr.Image(value=original_frame, visible=True))

    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.selected_enhancer = enhancer
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = auto_rotate
    roop.globals.subsample_size = int(upsample[:3])

    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    roop.globals.execution_threads = roop.globals.CFG.max_threads
    face_index = SELECTED_INPUT_FACE_INDEX
    if len(roop.globals.INPUT_FACESETS) <= face_index:
        face_index = 0

    options = ProcessOptions(get_processing_plugins(mask_engine, swap_model=swap_model),
                              roop.globals.distance_threshold, roop.globals.blend_ratio,
                              roop.globals.face_swap_mode, face_index, clip_text, mask_json or None, num_steps, roop.globals.subsample_size, show_face_area, restore_original_mouth,
                              use_3d_recon=use_3d_recon,
                              use_source_bank=use_source_bank,
                              use_frontalization=use_frontalization,
                              frontalization_threshold=frontalization_threshold,
                              swap_model=swap_model)
    # Store so FBF frame navigation can regenerate swap previews for arbitrary frames.
    _last_preview_options = options

    current_frame = live_swap(current_frame, options)
    if current_frame is None:
        _last_swapped_preview = None
        return (gr.Image(visible=True),
                gr.Slider(info=timeinfo),
                gr.Image(value=original_frame, visible=True))
    _last_swapped_preview = current_frame          # cache for mask editor (BGR numpy)
    return (gr.Image(value=util.convert_to_gradio(current_frame), visible=True),
            gr.Slider(info=timeinfo),
            gr.Image(value=original_frame, visible=True))

def map_mask_engine(selected_mask_engine, clip_text):
    if selected_mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
          mask_engine = None
    elif selected_mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    else:
        mask_engine = None
    return mask_engine


# ── Masking modal JavaScript ──────────────────────────────────────────────────
# Injected into the page <head> via gr.Blocks(head=MASKING_HEAD_JS) in main.py.
# Gradio 5 strips <script> tags from gr.HTML values, so all interactive JS must
# go through this mechanism. The modal is built entirely in JavaScript and
# appended to document.body so position:fixed is never trapped by a CSS transform.
MASKING_HEAD_JS = """
<script>
(function() {
  'use strict';

  /* ── Per-modal state (reset each open) ────────────────────────────── */
  var _mode = 'exclude', _brush = 20, _painting = false, _lx = 0, _ly = 0;
  var _zoom = 1.0, _panX = 0, _panY = 0;
  var _panning = false, _panSX = 0, _panSY = 0, _panOX = 0, _panOY = 0;
  var _bgImage = null, _swappedImage = null, _prevRafPending = false;
  var _pendingMaskJson = null;   /* mask JSON waiting to be restored once canvases are sized */
  var _targetStoreId = 'mask_json_store';  /* Gradio element ID that maskApply writes to */

  /* ── Frame-by-frame mode state ────────────────────────────────────── */
  var _fbfMode = false;           /* frame-by-frame mode enabled */
  var _fbfFrame = 1;              /* 1-based index of the frame currently being edited */
  var _fbfTotal = 1;              /* total frames (read from preview_frame_num slider max) */
  /* Per-frame mask storage: { "frameNum": { "facesetIdx": snapData }, ... }
     Only frames that have been explicitly saved appear.
     Serialised to mask_per_frame_store on Apply & Close. */
  var _fbfMasks = {};
  var _fbfFetchSeq = 0;  /* incremented on every fetch; lets callback detect stale responses */

  /* ── Multi-faceset state ──────────────────────────────────────────── */
  var _fbfFaceset = 0;            /* faceset index currently being edited */

  /* ── Public: called by the Gradio button click (fn=None, js="...") ── */
  window.maskToggle = function() {
    var modal = document.getElementById('roop-mask-modal');
    if (modal) { _closeModal(false); } else { _targetStoreId = 'mask_json_store'; _openModal(); }
  };

  /* ── Public: Frame Editor variant — uses per-frame crop stores, skips preview gate ── */
  window.maskToggleFrameEditor = function() {
    var modal = document.getElementById('roop-mask-modal');
    if (modal) { _closeModal(false); return; }
    _targetStoreId = 'fe_mask_json_store';
    var cropEl = document.querySelector('#fe_mask_face_crop_store textarea, #fe_mask_face_crop_store input');
    var swapEl = document.querySelector('#fe_mask_face_swap_crop_store textarea, #fe_mask_face_swap_crop_store input');
    _openModal(cropEl ? cropEl.value : '', swapEl ? swapEl.value : '', true);
  };

  /* ── Public: called when target media is removed — closes modal if open
     and resets state so no stale mask lingers for the next file. ─────── */
  window.maskReset = function() {
    var m = document.getElementById('roop-mask-modal');
    if (m) {
      m.remove();
      document.removeEventListener('keydown', _escHandler);
      _setToggleLabel(false);
    }
    _bgImage = null; _swappedImage = null;
    _pendingMaskJson = null;
    _fbfMode = false; _fbfFrame = 1; _fbfTotal = 1; _fbfMasks = {}; _fbfFaceset = 0;
  };

  /* ── Open ─────────────────────────────────────────────────────────── */
  /* faceCropUrlArg, swpCropUrlArg, skipGateCheck are used by maskToggleFrameEditor
     to pass pre-loaded image URLs and bypass the faceswap-tab DOM lookups. */
  function _openModal(faceCropUrlArg, swpCropUrlArg, skipGateCheck) {
    _mode = 'exclude'; _brush = 20; _painting = false;
    _zoom = 1.0; _panX = 0; _panY = 0; _panning = false;
    _prevRafPending = false;
    /* Initialise / preserve frame-by-frame state across re-opens.
       _fbfMasks persists across opens so saved per-frame masks are not lost. */
    _fbfMode = false;
    _fbfFrame = 1;
    /* Read total frames from the Gradio preview_frame_num slider */
    var sliderWrap = document.getElementById('preview_frame_num');
    var sliderEl   = sliderWrap ? sliderWrap.querySelector('input[type="range"]') : null;
    _fbfTotal = sliderEl ? Math.max(1, parseInt(sliderEl.max, 10) || 1) : 1;
    if (sliderEl) {
      var cur = parseInt(sliderEl.value, 10) || 1;
      _fbfFrame = Math.max(1, Math.min(_fbfTotal, cur));
    }

    var faceCropUrl, swpCropUrl;

    if (!skipGateCheck) {
      var wrap = document.getElementById('roop_preview_image');
      var previewImg = wrap ? wrap.querySelector('img') : null;
      if (!previewImg || !previewImg.src || previewImg.naturalWidth === 0) {
        alert('Please generate a preview first before editing the mask.');
        return;
      }
      /* swappedUrl = the current Gradio preview (face-swapped result).
         Used in the live preview panel as the base image. */
      var swappedUrl = previewImg.src;

      /* faceCropUrl = the canonical 512×512 face crop produced by Python's align_crop.
         The mask is painted in this coordinate system, so it always tracks the face
         perfectly through any head motion without needing any affine warp.
         Falls back to origUrl (full frame) if the crop isn't available yet. */
      var origWrap  = document.getElementById('roop_original_frame');
      var origImgEl = origWrap ? origWrap.querySelector('img') : null;
      var origUrl   = (origImgEl && origImgEl.naturalWidth > 0) ? origImgEl.src : swappedUrl;

      var cropStoreEl = document.querySelector('#mask_face_crop_store textarea, #mask_face_crop_store input');
      var faceCropDataUrl = cropStoreEl ? cropStoreEl.value : '';
      faceCropUrl = (faceCropDataUrl && faceCropDataUrl.startsWith('data:image')) ? faceCropDataUrl : origUrl;

      var swpCropStoreEl = document.querySelector('#mask_face_swap_crop_store textarea, #mask_face_swap_crop_store input');
      var swpCropDataUrl = swpCropStoreEl ? swpCropStoreEl.value : '';
      /* Live-preview base: swapped face crop when available, else source face crop.
         Falls back to origUrl only when no face crop was detected at all. */
      swpCropUrl = (swpCropDataUrl && swpCropDataUrl.startsWith('data:image')) ? swpCropDataUrl : faceCropUrl;
    } else {
      /* Frame Editor path: caller provides the image URLs directly. */
      faceCropUrl = faceCropUrlArg || '';
      swpCropUrl  = swpCropUrlArg  || faceCropUrlArg || '';
    }

    /* _bgImage = source face crop — the editor drawing surface.
       Painting on this ensures the mask is in canonical face-crop coordinates. */
    _bgImage = new Image();
    _bgImage.onload = function() { _schedulePreview(); };
    _bgImage.src = faceCropUrl;

    /* _swappedImage = swapped face crop — the live-preview base.
       The preview shows the swap result with mask overlays so the user can
       see exactly which face features are included / excluded. */
    _swappedImage = new Image();
    _swappedImage.onload = function() { _schedulePreview(); };
    _swappedImage.src = swpCropUrl;

    var storeEl = document.querySelector('#' + _targetStoreId + ' textarea, #' + _targetStoreId + ' input');
    var existJson = storeEl ? storeEl.value : '';
    /* Extract only the current faceset's mask from the (possibly multi-faceset) JSON */
    var existJsonForFaceset = _extractFacesetMask(existJson, _fbfFaceset);

    /* Read faceset count from Python-updated store */
    var countEl = document.querySelector('#mask_faceset_count_store textarea, #mask_faceset_count_store input');
    var _numFacesets = countEl ? (parseInt(countEl.value, 10) || 1) : 1;
    if (_numFacesets < 1) _numFacesets = 1;

    /* Build modal DOM ─────────────────────────────────────────────── */
    var modal = document.createElement('div');
    modal.id = 'roop-mask-modal';
    modal.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.9);z-index:2147483647;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif;';

    var panel = document.createElement('div');
    panel.style.cssText = [
      'background:#1c1c1c;border:1px solid #383838;border-radius:12px;',
      'padding:16px;width:96vw;height:94vh;',
      'display:flex;flex-direction:column;gap:8px;overflow:hidden;box-sizing:border-box;'
    ].join('');
    modal.appendChild(panel);

    panel.innerHTML = [
      /* ── Toolbar ── */
      '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;flex-shrink:0;">',
        '<button id="mask-btn-exclude" style="background:#3d1a1a;border:2px solid #f44336;color:#f44336;padding:6px 14px;border-radius:6px;cursor:pointer;font-weight:600;font-size:13px;">&#x1F534; Exclude</button>',
        '<button id="mask-btn-erase"   style="background:#1c1c1c;border:2px solid #383838;color:#999;padding:6px 14px;border-radius:6px;cursor:pointer;font-weight:600;font-size:13px;">&#x2B1C; Erase</button>',
        '<div style="width:1px;background:#383838;height:28px;margin:0 4px;"></div>',
        '<span style="color:#999;font-size:12px;">Brush:</span>',
        '<input type="range" id="mask-brush-sz" min="5" max="150" value="20" style="width:100px;accent-color:#50a070;cursor:pointer;vertical-align:middle;">',
        '<span id="mask-brush-lbl" style="color:#eee;font-size:12px;min-width:32px;">20px</span>',
        '<div style="width:1px;background:#383838;height:28px;margin:0 4px;"></div>',
        '<button id="mask-btn-zoom-out" title="Zoom out" style="background:#242424;border:1px solid #444;color:#ccc;padding:3px 10px;border-radius:5px;cursor:pointer;font-size:16px;font-weight:700;line-height:1.2;">&#x2212;</button>',
        '<span id="mask-zoom-lbl" style="color:#eee;font-size:12px;min-width:40px;text-align:center;">100%</span>',
        '<button id="mask-btn-zoom-in"  title="Zoom in"  style="background:#242424;border:1px solid #444;color:#ccc;padding:3px 10px;border-radius:5px;cursor:pointer;font-size:16px;font-weight:700;line-height:1.2;">+</button>',
        '<button id="mask-btn-zoom-rst" title="Reset zoom" style="background:#242424;border:1px solid #444;color:#aaa;padding:3px 9px;border-radius:5px;cursor:pointer;font-size:11px;">1:1</button>',
        '<div style="flex:1;"></div>',
        '<button id="mask-btn-clear"   style="background:#2c1010;border:1px solid #7a2020;color:#f08080;padding:6px 12px;border-radius:6px;cursor:pointer;font-size:13px;">&#x1F5D1; Clear</button>',
        '<button id="mask-btn-apply"   style="background:#3d8059;border:1px solid #50a070;color:#f0f0f0;padding:6px 14px;border-radius:6px;cursor:pointer;font-weight:600;font-size:13px;">&#x2705; Apply &amp; Close</button>',
        '<button id="mask-btn-discard" style="background:#242424;border:1px solid #383838;color:#bbb;padding:6px 12px;border-radius:6px;cursor:pointer;font-size:13px;">&#x2715; Discard</button>',
      '</div>',
      /* ── Legend row ── */
      '<div style="display:flex;gap:14px;font-size:11px;color:#888;flex-wrap:wrap;flex-shrink:0;align-items:center;">',
        '<span><span style="color:#f44336;font-size:14px;">&#x25A0;</span> Exclude &mdash; keep original</span>',
        '<span><span style="color:#aaa;font-size:14px;">&#x25A0;</span> Erase</span>',
        '<span style="color:#f0c040;font-size:11px;background:#2a2000;border:1px solid #5a4000;border-radius:4px;padding:2px 6px;">&#x26A0; Requires &ldquo;Face swap frames&rdquo; for preview</span>',
        '<span style="color:#555;font-size:11px;margin-left:auto;">Scroll=zoom &nbsp;|&nbsp; Middle-drag=pan &nbsp;|&nbsp; [Esc]=discard</span>',
      '</div>',
      /* ── Faceset selector row (hidden when only one source face loaded) ── */
      '<div id="mask-faceset-row" style="display:none;gap:8px;align-items:center;flex-shrink:0;',
           'background:#18181c;border:1px solid #383840;border-radius:8px;padding:5px 12px;">',
        '<span style="color:#888;font-size:11px;white-space:nowrap;">Masking source face:</span>',
        '<div id="mask-faceset-btns" style="display:flex;gap:4px;flex-wrap:wrap;"></div>',
      '</div>',
      /* ── Frame-by-frame mode row (hidden by default, shown when FBF toggle active) ── */
      '<div id="mask-fbf-row" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;flex-shrink:0;',
           'background:#181c24;border:1px solid #2a4060;border-radius:8px;padding:6px 12px;">',
        /* Toggle button */
        '<button id="mask-fbf-toggle" style="background:#1a2a40;border:1px solid #3a6090;color:#80b0e0;',
                'padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;white-space:nowrap;">',
          '&#x1F4FD; Frame-by-Frame: OFF',
        '</button>',
        '<div style="width:1px;background:#2a4060;height:24px;margin:0 4px;"></div>',
        /* Frame navigation — visible only when FBF is on */
        '<span id="mask-fbf-nav" style="display:none;gap:8px;align-items:center;">',
          '<button id="mask-fbf-prev" style="background:#242424;border:1px solid #444;color:#ccc;',
                  'padding:3px 10px;border-radius:5px;cursor:pointer;font-size:14px;">&#x25C4;</button>',
          '<span style="color:#aaa;font-size:12px;">Frame</span>',
          '<input id="mask-fbf-num" type="number" min="1" value="1"',
                 'style="width:58px;background:#111;border:1px solid #444;color:#eee;',
                        'border-radius:5px;padding:3px 6px;font-size:13px;text-align:center;">',
          '<span id="mask-fbf-total" style="color:#666;font-size:12px;">/ 1</span>',
          '<button id="mask-fbf-next" style="background:#242424;border:1px solid #444;color:#ccc;',
                  'padding:3px 10px;border-radius:5px;cursor:pointer;font-size:14px;">&#x25BA;</button>',
          '<div style="width:1px;background:#2a4060;height:24px;margin:0 4px;"></div>',
          '<button id="mask-fbf-save" style="background:#1a3d2a;border:1px solid #4CAF50;color:#4CAF50;',
                  'padding:4px 12px;border-radius:5px;cursor:pointer;font-size:12px;font-weight:600;">',
            '&#x1F4BE; Save for frame',
          '</button>',
          '<button id="mask-fbf-clear-frame" style="background:#2c1010;border:1px solid #7a2020;color:#f08080;',
                  'padding:4px 10px;border-radius:5px;cursor:pointer;font-size:12px;">',
            '&#x1F5D1; Clear frame',
          '</button>',
          '<div style="width:1px;background:#2a4060;height:24px;margin:0 4px;"></div>',
          '<input id="mask-fbf-range" type="text" placeholder="Also apply to frames: 1,5,10-20"',
                 'style="width:200px;background:#111;border:1px solid #444;color:#eee;',
                        'border-radius:5px;padding:3px 8px;font-size:12px;" title="Comma-separated frames or ranges, e.g. 1,3,5-10">',
          '<button id="mask-fbf-apply-range" style="background:#242424;border:1px solid #5a4000;color:#f0c040;',
                  'padding:4px 10px;border-radius:5px;cursor:pointer;font-size:12px;" title="Apply the current canvas to all specified frames">',
            '&#x2192; Apply to range',
          '</button>',
        '</span>',
        '<div style="flex:1;"></div>',
        /* Saved frames indicator */
        '<span id="mask-fbf-saved-lbl" style="color:#60a0d0;font-size:11px;font-style:italic;display:none;"></span>',
      '</div>',
      /* ── Two-column content area ── */
      '<div style="display:flex;gap:10px;flex:1;min-height:0;overflow:hidden;">',
        /* Editor column */
        '<div style="display:flex;flex-direction:column;gap:4px;flex:1;min-width:0;overflow:hidden;">',
          '<span style="color:#666;font-size:10px;font-weight:700;letter-spacing:.05em;flex-shrink:0;">EDITOR</span>',
          '<div id="mask-outer" style="flex:1;min-height:0;overflow:hidden;position:relative;border:1px solid #383838;border-radius:8px;cursor:none;background:#111;">',
            '<div id="mask-cvs-wrap" style="position:absolute;top:0;left:0;transform-origin:0 0;">',
              '<img id="mask-bg-img" style="display:block;user-select:none;" draggable="false">',
              '<canvas id="mask-cvs-exc" width="0" height="0" style="position:absolute;top:0;left:0;pointer-events:none;"></canvas>',
              '<canvas id="mask-cvs-inc" width="0" height="0" style="position:absolute;top:0;left:0;pointer-events:none;"></canvas>',
              '<canvas id="mask-cvs-cur" width="0" height="0" style="position:absolute;top:0;left:0;pointer-events:none;"></canvas>',
            '</div>',
          '</div>',
        '</div>',
        /* Live preview column */
        '<div style="display:flex;flex-direction:column;gap:4px;flex:1;min-width:0;overflow:hidden;">',
          '<span style="color:#666;font-size:10px;font-weight:700;letter-spacing:.05em;flex-shrink:0;">LIVE PREVIEW <span style="color:#444;font-weight:400;">(mask overlay)</span></span>',
          '<div style="flex:1;min-height:0;overflow:hidden;border:1px solid #383838;border-radius:8px;display:flex;align-items:center;justify-content:center;background:#111;">',
            '<canvas id="mask-preview-cvs" width="0" height="0" style="max-width:100%;max-height:100%;display:block;border-radius:6px;"></canvas>',
          '</div>',
        '</div>',
      '</div>'
    ].join('');

    document.body.appendChild(modal);

    /* ── Build faceset selector buttons ── */
    var facesetRow  = document.getElementById('mask-faceset-row');
    var facesetBtns = document.getElementById('mask-faceset-btns');
    if (_numFacesets > 1 && facesetRow && facesetBtns) {
      facesetRow.style.display = 'flex';
      for (var _fi = 0; _fi < _numFacesets; _fi++) {
        (function(idx) {
          var fb = document.createElement('button');
          fb.id = 'mask-fs-btn-' + idx;
          fb.textContent = 'Face ' + idx;
          fb.style.cssText = 'padding:4px 12px;border-radius:5px;cursor:pointer;font-size:12px;font-weight:600;border:2px solid #383838;color:#888;background:#1c1c1c;';
          fb.addEventListener('click', function() { _fbfSwitchFaceset(idx); });
          facesetBtns.appendChild(fb);
        })(_fi);
      }
      _fbfUpdateFacesetButtons();
    }

    /* Wire toolbar buttons */
    document.getElementById('mask-btn-exclude').addEventListener('click', function() { _setMode('exclude'); });
    document.getElementById('mask-btn-erase').addEventListener('click',   function() { _setMode('erase'); });
    document.getElementById('mask-btn-clear').addEventListener('click',   function() { _clearAll(); });
    document.getElementById('mask-btn-apply').addEventListener('click',   function() { maskApply(); });
    document.getElementById('mask-btn-discard').addEventListener('click', function() { _closeModal(false); });
    document.getElementById('mask-brush-sz').addEventListener('input',    function() { _setBrush(this.value); });
    document.getElementById('mask-btn-zoom-in').addEventListener('click',  function() { _zoomBy(1.25); });
    document.getElementById('mask-btn-zoom-out').addEventListener('click', function() { _zoomBy(0.8); });
    document.getElementById('mask-btn-zoom-rst').addEventListener('click', function() { _resetZoom(); });

    /* ── Frame-by-frame controls ── */
    document.getElementById('mask-fbf-toggle').addEventListener('click', _fbfToggle);
    document.getElementById('mask-fbf-prev').addEventListener('click',   function() { _fbfGoTo(_fbfFrame - 1); });
    document.getElementById('mask-fbf-next').addEventListener('click',   function() { _fbfGoTo(_fbfFrame + 1); });
    document.getElementById('mask-fbf-save').addEventListener('click',   _fbfSaveFrame);
    document.getElementById('mask-fbf-clear-frame').addEventListener('click', _fbfClearFrame);
    document.getElementById('mask-fbf-apply-range').addEventListener('click', _fbfApplyRange);
    document.getElementById('mask-fbf-num').addEventListener('change', function() {
      var n = parseInt(this.value, 10);
      if (!isNaN(n)) _fbfGoTo(n);
    });
    /* Update total-frame label now that we know _fbfTotal */
    var totalLbl = document.getElementById('mask-fbf-total');
    if (totalLbl) totalLbl.textContent = '/ ' + _fbfTotal;
    var numInput = document.getElementById('mask-fbf-num');
    if (numInput) { numInput.max = _fbfTotal; numInput.value = _fbfFrame; }

    /* Wire outer container events (draw + zoom + pan) */
    var outer = document.getElementById('mask-outer');
    outer.addEventListener('wheel',      _onWheel, { passive: false });
    outer.addEventListener('mousedown',  _onOuterMouseDown);
    outer.addEventListener('mousemove',  _onOuterMouseMove);
    outer.addEventListener('mouseup',    _onOuterMouseUp);
    outer.addEventListener('mouseleave', _onOuterMouseLeave);
    /* Touch */
    outer.addEventListener('touchstart', _onOuterTouchStart, { passive: false });
    outer.addEventListener('touchmove',  _onOuterTouchMove,  { passive: false });
    outer.addEventListener('touchend',   function() { _painting = false; });

    _setToggleLabel(true);

    var bgImg = document.getElementById('mask-bg-img');
    bgImg.onload = function() { requestAnimationFrame(_setupCanvas); };
    bgImg.src = faceCropUrl;  /* canonical face crop — editor background */
    requestAnimationFrame(_setupCanvas);

    /* Store the current-faceset mask to be restored inside _setupCanvas once canvases are sized.
       Calling _restoreMask here would race: the canvas images load async and check !c.width,
       which would be 0 at this point and silently bail. */
    _pendingMaskJson = existJsonForFaceset || null;
    document.addEventListener('keydown', _escHandler);
  }

  /* ── Canvas sizing ────────────────────────────────────────────────── */
  function _setupCanvas() {
    var modal = document.getElementById('roop-mask-modal');
    if (!modal || modal.dataset.canvasReady === '1') return;

    var img = document.getElementById('mask-bg-img');
    if (!img || !img.naturalWidth) { requestAnimationFrame(_setupCanvas); return; }

    var nw = img.naturalWidth, nh = img.naturalHeight;

    /* Available space = outer container minus a small margin */
    var outer = document.getElementById('mask-outer');
    var orect = outer ? outer.getBoundingClientRect() : null;
    var avW = (orect && orect.width  > 20) ? Math.floor(orect.width  - 4) : Math.floor(window.innerWidth  * 0.45);
    var avH = (orect && orect.height > 20) ? Math.floor(orect.height - 4) : Math.floor(window.innerHeight * 0.65);

    var s  = Math.min(1, avW / nw, avH / nh);
    var dw = Math.max(1, Math.floor(nw * s));
    var dh = Math.max(1, Math.floor(nh * s));

    if (!dw || !dh) { requestAnimationFrame(_setupCanvas); return; }

    modal.dataset.canvasReady = '1';
    modal.dataset.imgW = String(dw);
    modal.dataset.imgH = String(dh);

    img.style.width  = dw + 'px';
    img.style.height = dh + 'px';

    ['mask-cvs-exc', 'mask-cvs-inc', 'mask-cvs-cur'].forEach(function(id) {
      var c = document.getElementById(id); if (!c) return;
      c.width = dw; c.height = dh;
      c.style.width = dw + 'px'; c.style.height = dh + 'px';
    });

    var pc = document.getElementById('mask-preview-cvs');
    if (pc) { pc.width = dw; pc.height = dh; }

    _applyTransform();

    /* Restore any previously saved mask now that canvases have real dimensions.
       Images load asynchronously; their onload handlers will find c.width > 0. */
    if (_pendingMaskJson) {
      var json = _pendingMaskJson;
      _pendingMaskJson = null;
      _restoreMask(json);
    }

    _updatePreview();
  }

  /* ── Zoom / pan helpers ───────────────────────────────────────────── */
  function _applyTransform() {
    var wrap = document.getElementById('mask-cvs-wrap');
    if (!wrap) return;
    wrap.style.transform = 'translate(' + _panX + 'px,' + _panY + 'px) scale(' + _zoom + ')';
    var lbl = document.getElementById('mask-zoom-lbl');
    if (lbl) lbl.textContent = Math.round(_zoom * 100) + '%';
  }

  function _zoomBy(factor) {
    var outer = document.getElementById('mask-outer');
    var orect = outer ? outer.getBoundingClientRect() : null;
    _zoomAt(orect ? orect.width / 2 : 0, orect ? orect.height / 2 : 0, factor);
  }

  function _zoomAt(mx, my, factor) {
    var newZoom = Math.min(10, Math.max(0.2, _zoom * factor));
    _panX = mx - (mx - _panX) * (newZoom / _zoom);
    _panY = my - (my - _panY) * (newZoom / _zoom);
    _zoom = newZoom;
    _clampPan();
    _applyTransform();
  }

  function _resetZoom() {
    _zoom = 1.0; _panX = 0; _panY = 0; _applyTransform();
  }

  function _clampPan() {
    var modal = document.getElementById('roop-mask-modal');
    var dw = parseInt((modal && modal.dataset.imgW) || '0');
    var dh = parseInt((modal && modal.dataset.imgH) || '0');
    var outer = document.getElementById('mask-outer');
    var ow = outer ? outer.clientWidth  : 0;
    var oh = outer ? outer.clientHeight : 0;
    var sw = dw * _zoom, sh = dh * _zoom;
    var mg = 80; /* allow scroll this many px past the edge */
    _panX = Math.min(mg, Math.max(ow - sw - mg, _panX));
    _panY = Math.min(mg, Math.max(oh - sh - mg, _panY));
  }

  /* Convert outer-container-relative coords to canvas pixel coords */
  function _outerToCanvas(mx, my) {
    return { x: (mx - _panX) / _zoom, y: (my - _panY) / _zoom };
  }

  /* ── Outer container event handlers ──────────────────────────────── */
  function _onWheel(e) {
    e.preventDefault();
    var outer = document.getElementById('mask-outer');
    var orect = outer ? outer.getBoundingClientRect() : null; if (!orect) return;
    _zoomAt(e.clientX - orect.left, e.clientY - orect.top, e.deltaY < 0 ? 1.15 : (1 / 1.15));
    var p = _outerToCanvas(e.clientX - orect.left, e.clientY - orect.top);
    _drawCursor(p.x, p.y);
  }

  function _onOuterMouseDown(e) {
    if (e.button === 1) { /* middle = pan */
      e.preventDefault();
      _panning = true; _panSX = e.clientX; _panSY = e.clientY; _panOX = _panX; _panOY = _panY;
      return;
    }
    if (e.button === 0) {
      var outer = document.getElementById('mask-outer');
      var orect = outer ? outer.getBoundingClientRect() : null; if (!orect) return;
      var p = _outerToCanvas(e.clientX - orect.left, e.clientY - orect.top);
      _painting = true; _lx = p.x; _ly = p.y;
      _paint(p.x, p.y, p.x, p.y);
      _schedulePreview();
    }
  }

  function _onOuterMouseMove(e) {
    var outer = document.getElementById('mask-outer');
    var orect = outer ? outer.getBoundingClientRect() : null; if (!orect) return;
    var mx = e.clientX - orect.left, my = e.clientY - orect.top;
    if (_panning) {
      _panX = _panOX + (e.clientX - _panSX);
      _panY = _panOY + (e.clientY - _panSY);
      _clampPan(); _applyTransform(); return;
    }
    var p = _outerToCanvas(mx, my);
    _drawCursor(p.x, p.y);
    if (_painting) {
      _paint(_lx, _ly, p.x, p.y);
      _lx = p.x; _ly = p.y;
      _schedulePreview();
    }
  }

  function _onOuterMouseUp(e) {
    if (e.button === 1) { _panning = false; }
    if (e.button === 0) { _painting = false; }
  }

  function _onOuterMouseLeave() {
    _painting = false; _panning = false; _clearCursor();
  }

  function _onOuterTouchStart(e) {
    e.preventDefault();
    var t = e.touches[0];
    var outer = document.getElementById('mask-outer');
    var orect = outer ? outer.getBoundingClientRect() : null; if (!orect) return;
    var p = _outerToCanvas(t.clientX - orect.left, t.clientY - orect.top);
    _painting = true; _lx = p.x; _ly = p.y;
    _paint(p.x, p.y, p.x, p.y); _schedulePreview();
  }

  function _onOuterTouchMove(e) {
    e.preventDefault();
    var t = e.touches[0];
    var outer = document.getElementById('mask-outer');
    var orect = outer ? outer.getBoundingClientRect() : null; if (!orect) return;
    var p = _outerToCanvas(t.clientX - orect.left, t.clientY - orect.top);
    _drawCursor(p.x, p.y);
    if (_painting) { _paint(_lx, _ly, p.x, p.y); _lx = p.x; _ly = p.y; _schedulePreview(); }
  }

  /* ── Drawing helpers ──────────────────────────────────────────────── */
  function _setMode(m) {
    _mode = m;
    ['exclude', 'erase'].forEach(function(mm) {
      var b = document.getElementById('mask-btn-' + mm); if (!b) return;
      if (mm === m) {
        var col = mm === 'exclude' ? '#f44336' : '#cccccc';
        var bg  = mm === 'exclude' ? '#3d1a1a' : '#2c2c2c';
        b.style.borderColor = col; b.style.color = col; b.style.background = bg;
      } else {
        b.style.borderColor = '#383838'; b.style.color = '#999'; b.style.background = '#1c1c1c';
      }
    });
  }

  function _setBrush(v) {
    _brush = parseInt(v);
    var lbl = document.getElementById('mask-brush-lbl');
    if (lbl) lbl.textContent = v + 'px';
  }

  function _drawCursor(x, y) {
    var c = document.getElementById('mask-cvs-cur'); if (!c || !c.width) return;
    var ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    var col = _mode === 'exclude' ? '#f44336' : '#ffffff';
    ctx.beginPath(); ctx.arc(x, y, Math.max(_brush / 2, 2), 0, Math.PI * 2);
    ctx.strokeStyle = col; ctx.lineWidth = 2 / _zoom; ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 2 / _zoom, 0, Math.PI * 2);
    ctx.fillStyle = col; ctx.fill();
  }

  function _clearCursor() {
    var c = document.getElementById('mask-cvs-cur');
    if (c && c.width) c.getContext('2d').clearRect(0, 0, c.width, c.height);
  }

  function _paint(x1, y1, x2, y2) {
    if (_mode === 'erase') {
      _eraseOn('mask-cvs-exc', x1, y1, x2, y2);
      _eraseOn('mask-cvs-inc', x1, y1, x2, y2);
      return;
    }
    var c = document.getElementById('mask-cvs-exc'); if (!c || !c.width) return;
    var ctx = c.getContext('2d');
    ctx.globalCompositeOperation = 'source-over';
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = _brush;
    ctx.strokeStyle = 'rgba(244,67,54,0.65)';
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }

  function _eraseOn(cid, x1, y1, x2, y2) {
    var c = document.getElementById(cid); if (!c || !c.width) return;
    var ctx = c.getContext('2d');
    ctx.globalCompositeOperation = 'destination-out';
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.lineWidth = _brush; ctx.strokeStyle = 'rgba(0,0,0,1)';
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.globalCompositeOperation = 'source-over';
  }

  function _clearAll() {
    ['mask-cvs-exc', 'mask-cvs-inc'].forEach(function(id) {
      var c = document.getElementById(id);
      if (c && c.width) c.getContext('2d').clearRect(0, 0, c.width, c.height);
    });
    _updatePreview();
  }

  /* ── Live preview ─────────────────────────────────────────────────── */
  function _schedulePreview() {
    if (_prevRafPending) return;
    _prevRafPending = true;
    requestAnimationFrame(function() { _prevRafPending = false; _updatePreview(); });
  }

  function _updatePreview() {
    var pc = document.getElementById('mask-preview-cvs');
    if (!pc || !pc.width || !pc.height) return;
    var ctx = pc.getContext('2d');
    ctx.clearRect(0, 0, pc.width, pc.height);

    /* Step 1: Draw the face-swapped result as the base.
       Falls back to the original frame if swapped isn't loaded yet. */
    var base = (_swappedImage && _swappedImage.complete && _swappedImage.naturalWidth)
               ? _swappedImage
               : ((_bgImage && _bgImage.complete && _bgImage.naturalWidth) ? _bgImage : null);
    if (base) {
      ctx.drawImage(base, 0, 0, pc.width, pc.height);
    } else {
      ctx.fillStyle = '#1a1a1a'; ctx.fillRect(0, 0, pc.width, pc.height);
    }

    /* Step 2: Where the user painted Exclude → reveal the original source frame.
       This shows exactly what the mask will do: excluded pixels revert to original. */
    var excC = document.getElementById('mask-cvs-exc');
    var origReady = _bgImage && _bgImage.complete && _bgImage.naturalWidth;
    if (excC && excC.width && origReady) {
      /* Composite: original image, clipped to the exclude-painted region */
      var tmp = document.createElement('canvas');
      tmp.width = pc.width; tmp.height = pc.height;
      var tc = tmp.getContext('2d');
      tc.drawImage(_bgImage, 0, 0, pc.width, pc.height);  /* original pixels */
      tc.globalCompositeOperation = 'destination-in';
      tc.drawImage(excC, 0, 0, pc.width, pc.height);      /* clip to exclude strokes */
      ctx.drawImage(tmp, 0, 0);                           /* stamp over swapped base */
    }

    /* Note: Areas with no exclude paint show the swapped result from step 1.
       Only excluded regions show the original face. */
  }

  /* ── Mask serialisation ───────────────────────────────────────────── */
  function _toGray(canvas) {
    var w = canvas.width, h = canvas.height;
    var tmp = document.createElement('canvas'); tmp.width = w; tmp.height = h;
    var ctx = tmp.getContext('2d');
    var src = canvas.getContext('2d').getImageData(0, 0, w, h);
    var out = ctx.getImageData(0, 0, w, h);
    for (var i = 0; i < src.data.length; i += 4) {
      var a = src.data[i + 3];
      out.data[i] = a; out.data[i+1] = a; out.data[i+2] = a; out.data[i+3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    return tmp.toDataURL('image/png');
  }

  function _isBlank(c) {
    if (!c || !c.width || !c.height) return true;
    var blank = document.createElement('canvas'); blank.width = c.width; blank.height = c.height;
    return c.toDataURL() === blank.toDataURL();
  }

  function _restoreMask(jsonStr) {
    if (!jsonStr) return;
    try {
      var d = JSON.parse(jsonStr);

      /* _toGray() stored each canvas as a PNG where:
           R=G=B = original alpha value of the painted pixel  (0‒255)
           A     = 255 always (fully opaque)
         Unpainted pixels therefore come back as solid black (R=G=B=0, A=255).
         We must NOT drawImage() this directly onto the canvas — that would
         paint black over every transparent pixel and break further editing.
         Instead: read the brightness (R channel) back as the alpha, reconstruct
         the paint colour at that opacity, and leave zero-brightness pixels fully
         transparent so the editor background shows through correctly. */
      function loadLayer(url, cid) {
        if (!url) return;
        var tmpImg = new Image();
        tmpImg.onload = function() {
          var c = document.getElementById(cid); if (!c || !c.width) return;

          /* Decode the grayscale PNG into raw pixel data */
          var off = document.createElement('canvas');
          off.width = c.width; off.height = c.height;
          var octx = off.getContext('2d');
          octx.drawImage(tmpImg, 0, 0, c.width, c.height);
          var imgData = octx.getImageData(0, 0, c.width, c.height);
          var px = imgData.data;

          /* Paint colour for each layer (matches the original stroke colours) */
          var isInc = (cid === 'mask-cvs-inc');
          var pr = isInc ?  76 : 244;   /* include = green, exclude = red */
          var pg = isInc ? 175 :  67;
          var pb = isInc ?  80 :  54;

          /* Convert: brightness → alpha; fill with paint colour */
          for (var i = 0; i < px.length; i += 4) {
            var brightness = px[i]; /* R channel holds the original alpha */
            px[i]   = pr;
            px[i+1] = pg;
            px[i+2] = pb;
            px[i+3] = brightness;  /* 0 = transparent (unpainted), >0 = painted */
          }
          octx.putImageData(imgData, 0, 0);

          /* Draw the reconstructed paint layer onto the real canvas */
          c.getContext('2d').drawImage(off, 0, 0);
          _schedulePreview();
        };
        tmpImg.src = url;
      }

      loadLayer(d.exclude, 'mask-cvs-exc');
      loadLayer(d.include, 'mask-cvs-inc');
    } catch(e) {}
  }

  /* ── Frame-by-frame helpers ───────────────────────────────────────── */

  function _fbfUpdateSavedLabel() {
    var lbl = document.getElementById('mask-fbf-saved-lbl');
    if (!lbl) return;
    /* Show saved frames for the current faceset */
    var fKey = String(_fbfFaceset);
    var keys = Object.keys(_fbfMasks).filter(function(k) {
      return _fbfMasks[k] && _fbfMasks[k][fKey];
    });
    if (keys.length === 0) {
      lbl.textContent = '';
      lbl.style.display = 'none';
    } else {
      /* Show compact sorted list, collapsing runs into ranges */
      var nums = keys.map(Number).sort(function(a,b){return a-b;});
      var parts = [], i = 0;
      while (i < nums.length) {
        var start = nums[i], end = start;
        while (i + 1 < nums.length && nums[i+1] === nums[i] + 1) { i++; end = nums[i]; }
        parts.push(end > start ? start + '–' + end : String(start));
        i++;
      }
      var suffix = (document.getElementById('mask-fs-btn-1')) ? ' [Face ' + _fbfFaceset + ']' : '';
      lbl.textContent = '✔ Custom masks: frames ' + parts.join(', ') + suffix;
      lbl.style.display = 'inline';
    }
  }

  /* ── Faceset-mask helpers ─────────────────────────────────────────── */

  function _extractFacesetMask(jsonStr, facesetIdx) {
    /* Given a (possibly multi-faceset) mask JSON string, return just the
       single-faceset flat object {exclude, include, canonical} for facesetIdx,
       serialised back to a JSON string.  Returns '' when nothing is stored. */
    if (!jsonStr) return '';
    try {
      var obj = JSON.parse(jsonStr);
      var fKey = String(facesetIdx);
      /* New multi-faceset format: all top-level keys are digit strings */
      var keys = Object.keys(obj);
      var isNew = keys.length > 0 && keys.every(function(k) { return /^\d+$/.test(k); });
      if (isNew) {
        var entry = obj[fKey];
        return entry ? JSON.stringify(entry) : '';
      }
      /* Old flat format (single-faceset): treat as faceset 0 */
      return (facesetIdx === 0) ? jsonStr : '';
    } catch(e) { return ''; }
  }

  function _fbfUpdateFacesetButtons() {
    /* Highlight the currently-active faceset button, dim the rest */
    var countEl = document.querySelector('#mask_faceset_count_store textarea, #mask_faceset_count_store input');
    var numFacesets = countEl ? (parseInt(countEl.value, 10) || 1) : 1;
    for (var i = 0; i < numFacesets; i++) {
      var btn = document.getElementById('mask-fs-btn-' + i);
      if (!btn) continue;
      if (i === _fbfFaceset) {
        btn.style.borderColor = '#50a070'; btn.style.color = '#90e0b0'; btn.style.background = '#1a3a2a';
      } else {
        btn.style.borderColor = '#383838'; btn.style.color = '#888'; btn.style.background = '#1c1c1c';
      }
    }
  }

  function _fbfSetCurrentFacesetMask(snap) {
    /* Save snap into _fbfMasks[_fbfFrame][_fbfFaceset] */
    var frameKey = String(_fbfFrame);
    var fKey = String(_fbfFaceset);
    if (!_fbfMasks[frameKey]) _fbfMasks[frameKey] = {};
    _fbfMasks[frameKey][fKey] = snap;
  }

  function _fbfSwitchFaceset(idx) {
    if (idx === _fbfFaceset) return;
    /* Save current canvas before switching so work isn't lost */
    if (_fbfMode) {
      _fbfSetCurrentFacesetMask(_fbfSnapshotCanvas());
    }
    _fbfFaceset = idx;
    _fbfUpdateFacesetButtons();
    /* Restore the mask for this faceset */
    if (_fbfMode) {
      var key = String(_fbfFrame);
      var fKey = String(_fbfFaceset);
      var snap = (_fbfMasks[key] && _fbfMasks[key][fKey]) ? _fbfMasks[key][fKey] : null;
      _fbfRestoreCanvas(snap);
    } else {
      /* Not in FBF mode: restore the global mask for the new faceset */
      var storeEl = document.querySelector('#' + _targetStoreId + ' textarea, #' + _targetStoreId + ' input');
      var jsonStr = storeEl ? storeEl.value : '';
      var facesetJson = _extractFacesetMask(jsonStr, _fbfFaceset);
      _fbfRestoreCanvas(facesetJson ? JSON.parse(facesetJson) : null);
    }
    _fbfUpdateSavedLabel();
    /* Fetch new face crop for this faceset on the current frame */
    _fbfFetchFaceCrop(_fbfFrame);
  }

  function _fbfToggle() {
    _fbfMode = !_fbfMode;
    var btn = document.getElementById('mask-fbf-toggle');
    var nav = document.getElementById('mask-fbf-nav');
    if (_fbfMode) {
      if (btn) { btn.textContent = '📽 Frame-by-Frame: ON'; btn.style.background = '#1a3060'; btn.style.borderColor = '#4080c0'; btn.style.color = '#80c0ff'; }
      if (nav) nav.style.display = 'inline-flex';
      _fbfUpdateSavedLabel();
    } else {
      if (btn) { btn.textContent = '📽 Frame-by-Frame: OFF'; btn.style.background = '#1a2a40'; btn.style.borderColor = '#3a6090'; btn.style.color = '#80b0e0'; }
      if (nav) nav.style.display = 'none';
    }
  }

  function _fbfSnapshotCanvas() {
    /* Capture the current canvas state as {include, exclude} PNG data-URLs */
    var excC = document.getElementById('mask-cvs-exc');
    var incC = document.getElementById('mask-cvs-inc');
    var snap = {};
    if (!_isBlank(excC)) snap.exclude = _toGray(excC);
    if (!_isBlank(incC)) snap.include = _toGray(incC);
    snap.canonical = true;
    return snap;
  }

  function _fbfRestoreCanvas(snap) {
    _clearAll();
    if (!snap) return;
    /* Reuse _restoreMask's loadLayer logic — it handles the grayscale→paint decode */
    _restoreMask(JSON.stringify(snap));
  }

  function _fbfGoTo(n) {
    n = Math.max(1, Math.min(_fbfTotal, n));
    if (n === _fbfFrame && _fbfMode) {
      /* Same frame — just sync the input */
      var inp = document.getElementById('mask-fbf-num');
      if (inp) inp.value = n;
      return;
    }
    _fbfFrame = n;
    var inp = document.getElementById('mask-fbf-num');
    if (inp) inp.value = n;
    /* Load saved mask for the destination frame + current faceset (instant, no server round-trip) */
    var key = String(n);
    var fKey = String(_fbfFaceset);
    var snap = (_fbfMasks[key] && _fbfMasks[key][fKey]) ? _fbfMasks[key][fKey] : null;
    _fbfRestoreCanvas(snap);
    _fbfUpdateSavedLabel();
    /* Fetch the face crop for this frame from Python and update the editor background */
    _fbfFetchFaceCrop(n);
  }

  function _fbfSaveFrame() {
    _fbfSetCurrentFacesetMask(_fbfSnapshotCanvas());
    _fbfUpdateSavedLabel();
    /* Flash the save button green briefly */
    var btn = document.getElementById('mask-fbf-save');
    if (btn) {
      var orig = btn.style.background;
      btn.style.background = '#0a5a28';
      setTimeout(function() { if (btn) btn.style.background = orig; }, 500);
    }
  }

  function _fbfClearFrame() {
    var key = String(_fbfFrame);
    var fKey = String(_fbfFaceset);
    if (_fbfMasks[key]) {
      delete _fbfMasks[key][fKey];
      /* Remove the frame entry entirely if no facesets remain */
      if (Object.keys(_fbfMasks[key]).length === 0) delete _fbfMasks[key];
    }
    _clearAll();
    _fbfUpdateSavedLabel();
  }

  function _fbfParseRange(str) {
    /* Parse "1,3,5-10,20" into an array of frame numbers */
    var result = [];
    (str || '').split(',').forEach(function(part) {
      part = part.trim();
      var m = part.match(/^(\d+)\s*[-–]\s*(\d+)$/);
      if (m) {
        var lo = parseInt(m[1], 10), hi = parseInt(m[2], 10);
        for (var i = lo; i <= hi; i++) result.push(i);
      } else {
        var n = parseInt(part, 10);
        if (!isNaN(n)) result.push(n);
      }
    });
    return result;
  }

  function _fbfApplyRange() {
    var input = document.getElementById('mask-fbf-range');
    var rangeStr = input ? input.value : '';
    if (!rangeStr.trim()) { alert('Enter frame numbers or ranges, e.g. 1,5,10-20'); return; }
    var frames = _fbfParseRange(rangeStr);
    if (frames.length === 0) { alert('No valid frame numbers found.'); return; }
    var snap = _fbfSnapshotCanvas();
    var fKey = String(_fbfFaceset);
    frames.forEach(function(n) {
      if (n >= 1 && n <= _fbfTotal) {
        if (!_fbfMasks[String(n)]) _fbfMasks[String(n)] = {};
        _fbfMasks[String(n)][fKey] = JSON.parse(JSON.stringify(snap));
      }
    });
    if (input) input.value = '';
    _fbfUpdateSavedLabel();
    alert('Mask applied to ' + frames.length + ' frame(s).');
  }

  /* ── FBF face-crop refresh ────────────────────────────────────────── */
  /* Called each time the user navigates to a new frame in FBF mode.
     Writes "frameNum:seq" to fbf_frame_num_store.  The ":seq" suffix ensures
     the value always changes (Gradio's .change() won't fire if the value is
     identical to the previous write, e.g. navigating back to the same frame).
     _writeToDedicatedStore dispatches an input event which Svelte picks up and
     propagates to Gradio's .change() handler → Python → .then() → _fbfOnCropReady.
     No hidden button required. */
  function _fbfFetchFaceCrop(frameNum) {
    var modal = document.getElementById('roop-mask-modal');
    if (!modal) return;

    /* Increment sequence counter so any prior in-flight callback knows it's stale */
    var seq = ++_fbfFetchSeq;

    /* Show a loading indicator */
    var bgImg = document.getElementById('mask-bg-img');
    if (bgImg) {
      bgImg.style.opacity = '0.4';
      bgImg.title = 'Loading frame ' + frameNum + '…';
    }

    /* One-shot callback — Gradio 5's .then(fn=None, js=...) does NOT pass Python
       return values as JS arguments, so we read them directly from the DOM stores
       that Gradio has already updated by the time .then() fires. */
    window._fbfOnCropReady = function() {
      if (seq !== _fbfFetchSeq) {
        console.log('[FBF] _fbfOnCropReady stale seq', seq, '!==', _fbfFetchSeq, '— discarding');
        return;
      }
      window._fbfOnCropReady = null;
      /* Read the updated values straight from the DOM — Svelte has flushed by now */
      var cropEl = document.querySelector('#mask_face_crop_store textarea, #mask_face_crop_store input');
      var swapEl = document.querySelector('#mask_face_swap_crop_store textarea, #mask_face_swap_crop_store input');
      var cropUrl = cropEl ? cropEl.value : '';
      var swapUrl = swapEl ? swapEl.value : '';
      console.log('[FBF] _fbfOnCropReady: DOM cropUrl len=', cropUrl.length, ' swapUrl len=', swapUrl.length);
      if (bgImg) { bgImg.style.opacity = '1'; bgImg.title = ''; }
      if (cropUrl && cropUrl.startsWith('data:image')) {
        var swp = (swapUrl && swapUrl.startsWith('data:image')) ? swapUrl : cropUrl;
        _fbfApplyNewCrops(cropUrl, swp, bgImg);
      }
    };

    /* Write "frameNum:seq" — the colon-suffix guarantees the value changes every
       call so Gradio's .change() always fires, even for repeated same-frame nav. */
    console.log('[FBF] writing fbf_frame_num_store =', frameNum + ':' + _fbfFaceset + ':' + seq);
    _writeToDedicatedStore('fbf_frame_num_store', frameNum + ':' + _fbfFaceset + ':' + seq);
  }

  function _fbfApplyNewCrops(faceCropUrl, swpCropUrl, bgImgEl) {
    /* Update _bgImage and _swappedImage with the new face crops, then redraw. */
    _bgImage = new Image();
    _bgImage.onload = function() { _schedulePreview(); };
    _bgImage.src = faceCropUrl;

    _swappedImage = new Image();
    _swappedImage.onload = function() { _schedulePreview(); };
    _swappedImage.src = swpCropUrl;

    /* Update the visible background in the editor */
    if (bgImgEl) {
      bgImgEl.style.opacity = '1';
      bgImgEl.title = '';
      /* Swap the src; the canvas stays the same size (512×512 face crops are consistent) */
      bgImgEl.src = faceCropUrl;
    }
    _schedulePreview();
  }

  function _writeToDedicatedStore(elemId, jstr) {
    var wrap = document.querySelector('#' + elemId);
    if (!wrap) return;
    var ta = wrap.querySelector('textarea') || wrap.querySelector('input[type="text"]');
    if (!ta) return;
    try {
      var setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(ta), 'value').set;
      setter.call(ta, jstr);
    } catch(ex) { ta.value = jstr; }
    ta.dispatchEvent(new Event('input',  { bubbles: true }));
    ta.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function _writeToStore(jstr) {
    _writeToDedicatedStore(_targetStoreId, jstr);
  }

  /* ── Apply ────────────────────────────────────────────────────────── */
  window.maskApply = function() {
    var excC = document.getElementById('mask-cvs-exc');
    var incC = document.getElementById('mask-cvs-inc');
    var result = {};
    if (!_isBlank(excC)) result.exclude = _toGray(excC);
    if (!_isBlank(incC)) result.include = _toGray(incC);
    /* Mark as canonical: mask was painted on the face-crop background,
       so ProcessMgr can apply it directly without any affine warp. */
    result.canonical = true;

    /* In frame-by-frame mode: auto-save the current canvas to the current frame,
       then persist the full per-frame map to mask_per_frame_store.
       The global mask_json_store is also written so single-frame previews work. */
    if (_fbfMode) {
      _fbfSetCurrentFacesetMask(_fbfSnapshotCanvas());
      _writeToDedicatedStore('mask_per_frame_store', JSON.stringify(_fbfMasks));
    }

    /* Build per-faceset global mask JSON:
       Read whatever is already in mask_json_store and merge in the current faceset.
       This preserves masks painted for OTHER facesets in previous Apply calls. */
    var storeEl = document.querySelector('#' + _targetStoreId + ' textarea, #' + _targetStoreId + ' input');
    var existJson = storeEl ? storeEl.value : '';
    var allMasks = {};
    if (existJson) {
      try {
        var existing = JSON.parse(existJson);
        var exKeys = Object.keys(existing);
        var isNew = exKeys.length > 0 && exKeys.every(function(k) { return /^\d+$/.test(k); });
        if (isNew) {
          allMasks = existing;
        } else {
          /* Old flat format — treat as faceset 0 */
          allMasks['0'] = existing;
        }
      } catch(e) {}
    }
    /* Write (or remove) the current faceset's mask */
    if (result.exclude || result.include) {
      allMasks[String(_fbfFaceset)] = result;
    } else {
      /* Canvas was fully erased — remove this faceset's entry */
      delete allMasks[String(_fbfFaceset)];
    }

    var wasFrameEditor = (_targetStoreId === 'fe_mask_json_store');
    _writeToStore(JSON.stringify(allMasks));
    _closeModal(false);
    /* Only auto-trigger the faceswap-tab refresh preview when NOT in Frame Editor mode.
       In Frame Editor mode the mask is stored and applied only on compile. */
    if (!wasFrameEditor) {
      setTimeout(function() {
        var wrap = document.getElementById('btn_refresh_preview');
        var btn  = wrap ? wrap.querySelector('button') : null;
        if (btn) btn.click();
      }, 150);
    }
  };

  /* ── Close ────────────────────────────────────────────────────────── */
  function _closeModal(_save) {
    _targetStoreId = 'mask_json_store';  /* reset to default after every close */
    var m = document.getElementById('roop-mask-modal');
    if (m) m.remove();
    _bgImage = null; _swappedImage = null;
    document.removeEventListener('keydown', _escHandler);
    _setToggleLabel(false);
  }

  function _escHandler(e) {
    if (e.key === 'Escape') {
      var m = document.getElementById('roop-mask-modal');
      if (m) _closeModal(false);
    }
  }

  function _setToggleLabel(active) {
    var btn = document.querySelector('#btn_toggle_masking button');
    if (!btn) return;
    btn.textContent = active ? '\\u2705 Masking Active \\u2014 click to close' : '\\uD83C\\uDFAD Edit Mask';
  }

})();
</script>
"""

def gen_processing_text(start, end):
    return f'Processing frame range [{start} - {end}]'

def on_set_frame(sender:str, frame_num):
    global selected_preview_index, list_files_process
    
    idx = selected_preview_index
    if list_files_process[idx].endframe == 0:
        return gen_processing_text(0,0)
    
    start = list_files_process[idx].startframe
    end = list_files_process[idx].endframe
    if sender.lower().endswith('start'):
        list_files_process[idx].startframe = min(frame_num, end)
    else:
        list_files_process[idx].endframe = max(frame_num, start)
    
    return gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)



def on_clear_input_faces():
    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    return ui.globals.ui_input_thumbs, "0"

def on_clear_destfiles():
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs.clear()
    # Also clear the manual mask — it belongs to the removed media.
    # Reset preview_frame_num to max=1 so stale cached values don't cause
    # a Gradio bounds error on the next event that includes the slider.
    return ui.globals.ui_target_thumbs, "", gr.Slider(value=1, maximum=1, info='0:00:00'), ''


def index_of_no_face_action(dropdown_text):
    global no_face_choices

    return no_face_choices.index(dropdown_text) 

def translate_swap_mode(dropdown_text):
    if dropdown_text == "Selected face":
        return "selected"
    elif dropdown_text == "First found":
        return "first"
    elif dropdown_text == "All input faces":
        return "all_input"
    elif dropdown_text == "All female":
        return "all_female"
    elif dropdown_text == "All male":
        return "all_male"
    
    return "all"


def start_swap( output_method, enhancer, detection, keep_frames, wait_after_extraction, skip_audio, face_distance, blend_ratio,
                selected_mask_engine, clip_text, processing_method, no_face_action, vr_mode, autorotate, restore_original_mouth, num_swap_steps, upsample, mask_json,
                use_3d_recon=False, mask_per_frame_json="",
                use_source_bank=False, use_frontalization=False,
                frontalization_threshold=25.0, swap_model='inswapper',
                progress=gr.Progress()):
    from ui.main import prepare_environment
    from roop.core import batch_process_regular
    global is_processing, list_files_process

    if list_files_process is None or len(list_files_process) <= 0:
        return gr.Button(variant="primary"), None
    
    if roop.globals.CFG.clear_output:
        shutil.rmtree(roop.globals.output_path)

    if not util.is_installed("ffmpeg"):
        msg = "ffmpeg is not installed! No video processing possible."
        gr.Warning(msg)

    prepare_environment()

    roop.globals.selected_enhancer = enhancer
    roop.globals.target_path = None
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.keep_frames = keep_frames
    roop.globals.wait_after_extraction = wait_after_extraction
    roop.globals.skip_audio = skip_audio
    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = autorotate
    roop.globals.subsample_size = int(upsample[:3])
    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    if roop.globals.face_swap_mode == 'selected':
        if len(roop.globals.TARGET_FACES) < 1:
            gr.Error('No Target Face selected!')
            return gr.Button(variant="primary"), None

    is_processing = True
    yield gr.Button(variant="secondary", interactive=False), gr.Button(variant="primary", interactive=True)
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None

    batch_process_regular(output_method, list_files_process, mask_engine, clip_text, processing_method == "In-Memory processing", mask_json or None, restore_original_mouth, num_swap_steps, progress, SELECTED_INPUT_FACE_INDEX,
                          use_3d_recon=use_3d_recon,
                          mask_per_frame_json=mask_per_frame_json or "",
                          use_source_bank=use_source_bank,
                          use_frontalization=use_frontalization,
                          frontalization_threshold=frontalization_threshold,
                          swap_model=swap_model)
    is_processing = False
    yield gr.Button(variant="primary", interactive=True), gr.Button(variant="secondary", interactive=False)


def stop_swap():
    roop.globals.processing = False
    gr.Info('Aborting processing - please wait for the remaining threads to be stopped')
    return gr.Button(variant="primary", interactive=True), gr.Button(variant="secondary", interactive=False)


def on_destfiles_changed(destfiles):
    global selected_preview_index, list_files_process, current_video_fps

    list_files_process.clear()
    if destfiles is None or len(destfiles) < 1:
        return gr.Slider(value=1, maximum=1, info='0:00:00'), ''

    for f in destfiles:
        list_files_process.append(ProcessEntry(f.name, 0,0, 0))

    selected_preview_index = 0
    idx = selected_preview_index    
    
    filename = list_files_process[idx].filename
    
    if util.is_video(filename) or filename.lower().endswith('gif') or util.is_animated_webp(filename):
        total_frames = get_video_frame_total(filename)
        if total_frames is None or total_frames < 1:
            total_frames = 1
            gr.Warning(f"Corrupted video {filename}, can't detect number of frames!")
        else:
            current_video_fps = util.detect_fps(filename) if not filename.lower().endswith('.webp') else 15
    else:
        total_frames = 1
    list_files_process[idx].endframe = total_frames
    if total_frames > 1:
        return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)
    return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), ''


def on_destfiles_selected(evt: gr.SelectData):
    global selected_preview_index, list_files_process, current_video_fps

    if evt is not None:
        selected_preview_index = evt.index
    idx = selected_preview_index
    filename = list_files_process[idx].filename
    if util.is_video(filename) or filename.lower().endswith('gif') or util.is_animated_webp(filename):
        total_frames = get_video_frame_total(filename)
        current_video_fps = util.detect_fps(filename) if not filename.lower().endswith('.webp') else 15
        if list_files_process[idx].endframe == 0:
            list_files_process[idx].endframe = total_frames
    else:
        total_frames = 1

    if total_frames > 1:
        return gr.Slider(value=list_files_process[idx].startframe, maximum=total_frames, info='0:00:00'), gen_processing_text(list_files_process[idx].startframe, list_files_process[idx].endframe)
    return gr.Slider(value=1, maximum=total_frames, info='0:00:00'), gen_processing_text(0, 0)


def get_gradio_output_format():
    if roop.globals.CFG.output_image_format == "jpg":
        return "jpeg"
    return roop.globals.CFG.output_image_format
