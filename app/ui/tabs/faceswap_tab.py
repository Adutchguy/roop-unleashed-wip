import os
import shutil
import gradio as gr
import roop.utilities as util
import roop.globals
import ui.globals
from roop.face_util import extract_face_images, create_blank_image
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

manual_masking = False


def faceswap_tab():
    global no_face_choices, previewimage

    with gr.Tab("🎭 Face Swap"):
        with gr.Row(variant='panel'):
            bt_srcfiles = gr.Files(label='Source Images or Facesets', file_count="multiple", file_types=["image", ".fsz"], elem_id='filelist', height=233)
            bt_destfiles = gr.Files(label='Target File(s)', file_count="multiple", file_types=["image", "video"], elem_id='filelist', height=233)
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
                            "Toggle manual masking", variant="secondary", size="sm"
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
                        bt_preview_mask = gr.Button(
                            "👥 Show Mask Preview", variant="secondary"
                        )

            with gr.Column(scale=2):
                previewimage = gr.Image(label="Preview Image", height=576, interactive=False, visible=True, format=get_gradio_output_format())
                maskimage = gr.ImageEditor(label="Manual mask Image", sources=["clipboard"], transforms="", type="numpy",
                                             brush=gr.Brush(color_mode="fixed", colors=["rgba(255, 255, 255, 1"]), interactive=True, visible=False)
                with gr.Row(variant='panel'):
                    fake_preview = gr.Checkbox(label="Face swap frames", value=False)
                    bt_refresh_preview = gr.Button("🔄 Refresh", variant='secondary', size='sm')
                    bt_use_face_from_preview = gr.Button("Use Face from this Frame", variant='primary', size='sm')
                with gr.Row():
                    preview_frame_num = gr.Slider(1, 1, value=1, label="Frame Number", info='0:00:00', step=1.0, interactive=True)
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
                        max_face_distance, ui.globals.ui_blend_ratio, selected_mask_engine, clip_text, no_face_action, vr_mode, autorotate, maskimage, chk_showmaskoffsets, chk_restoreoriginalmouth, num_swap_steps, ui.globals.ui_upscale]
    previewoutputs = [previewimage, maskimage, preview_frame_num] 
    input_faces.select(on_select_input_face, None, None).success(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)
    
    bt_move_left_input.click(fn=move_selected_input, inputs=[bt_move_left_input], outputs=[input_faces])
    bt_move_right_input.click(fn=move_selected_input, inputs=[bt_move_right_input], outputs=[input_faces])
    bt_move_left_target.click(fn=move_selected_target, inputs=[bt_move_left_target], outputs=[target_faces])
    bt_move_right_target.click(fn=move_selected_target, inputs=[bt_move_right_target], outputs=[target_faces])

    bt_remove_selected_input_face.click(fn=remove_selected_input_face, outputs=[input_faces])
    bt_srcfiles.upload(fn=on_srcfile_changed, show_progress='full', inputs=bt_srcfiles, outputs=[input_faces, bt_srcfiles])

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
    bt_destfiles.clear(fn=on_clear_destfiles, outputs=[target_faces])
    bt_clear_input_faces.click(fn=on_clear_input_faces, outputs=[input_faces])

    bt_preview_mask.click(fn=on_preview_mask, inputs=[preview_frame_num, bt_destfiles, clip_text, selected_mask_engine], outputs=[previewimage]) 

    start_event = bt_start.click(fn=start_swap,
        inputs=[output_method, ui.globals.ui_selected_enhancer, selected_face_detection, roop.globals.keep_frames, roop.globals.wait_after_extraction,
                    roop.globals.skip_audio, max_face_distance, ui.globals.ui_blend_ratio, selected_mask_engine, clip_text,video_swapping_method, no_face_action, vr_mode, autorotate, chk_restoreoriginalmouth, num_swap_steps, ui.globals.ui_upscale, maskimage],
        outputs=[bt_start, bt_stop], show_progress='full')

    bt_stop.click(fn=stop_swap, cancels=[start_event], outputs=[bt_start, bt_stop], queue=False)

    bt_refresh_preview.click(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)            
    bt_toggle_masking.click(fn=on_toggle_masking, inputs=[previewimage, maskimage], outputs=[previewimage, maskimage])            
    fake_preview.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs)
    preview_frame_num.release(fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs, show_progress='hidden', )
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
        while len(offs) < 10:
            offs.append(1.0)
        offs[index] = mask_offset
        roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = offs

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
                        face.mask_offsets = [0,0,0,0,20.0,10.0,1.0,1.0,1.0,1.0]
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
                face.mask_offsets = [0,0,0,0,20.0,10.0,1.0,1.0,1.0,1.0]
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
    return ui.globals.ui_input_thumbs, None


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

    return ui.globals.ui_input_thumbs

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
    elif util.is_video(roop.globals.target_path) or roop.globals.target_path.lower().endswith(('gif')):
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


def on_preview_frame_changed(frame_num, files, fake_preview, enhancer, detection, face_distance, blend_ratio,
                              selected_mask_engine, clip_text, no_face_action, vr_mode, auto_rotate, maskimage, show_face_area, restore_original_mouth, num_steps, upsample):
    global SELECTED_INPUT_FACE_INDEX, manual_masking, current_video_fps

    from roop.core import live_swap, get_processing_plugins

    manual_masking = False
    mask_offsets = [0,0,0,0,20.0,10.0,1.0,1.0,1.0,1.0]
    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        if not hasattr(roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0], 'mask_offsets'):
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = list(mask_offsets)
        mask_offsets = roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets
        while len(mask_offsets) < 10:
            mask_offsets.append(1.0)

    timeinfo = '0:00:00'
    if files is None or selected_preview_index >= len(files) or frame_num is None:
        return None,None, gr.Slider(info=timeinfo)

    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
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
        return None, None, gr.Slider(info=timeinfo)
    
    layers = None
    if maskimage is not None:
        layers = maskimage["layers"]

    if not fake_preview or len(roop.globals.INPUT_FACESETS) < 1:
        return gr.Image(value=util.convert_to_gradio(current_frame), visible=True), gr.ImageEditor(visible=False), gr.Slider(info=timeinfo)

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
    mask = layers[0] if layers is not None else None
    face_index = SELECTED_INPUT_FACE_INDEX
    if len(roop.globals.INPUT_FACESETS) <= face_index:
        face_index = 0
   
    options = ProcessOptions(get_processing_plugins(mask_engine), roop.globals.distance_threshold, roop.globals.blend_ratio,
                              roop.globals.face_swap_mode, face_index, clip_text, maskimage, num_steps, roop.globals.subsample_size, show_face_area, restore_original_mouth)

    current_frame = live_swap(current_frame, options)
    if current_frame is None:
        return gr.Image(visible=True), None, gr.Slider(info=timeinfo)
    return gr.Image(value=util.convert_to_gradio(current_frame), visible=True), gr.ImageEditor(visible=False), gr.Slider(info=timeinfo)

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


def on_toggle_masking(previewimage, mask):
    global manual_masking

    manual_masking = not manual_masking
    if manual_masking:
        layers = mask["layers"]
        if len(layers) == 1:
            layers = [create_blank_image(previewimage.shape[1],previewimage.shape[0])]
        return gr.Image(visible=False), gr.ImageEditor(value={"background": previewimage, "layers": layers, "composite": None}, visible=True)
    return gr.Image(visible=True), gr.ImageEditor(visible=False)

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


def on_preview_mask(frame_num, files, clip_text, mask_engine):
    from roop.core import live_swap, get_processing_plugins
    global is_processing

    if is_processing or files is None or selected_preview_index >= len(files) or clip_text is None or frame_num is None:
        return None
        
    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num
                                        )
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None or mask_engine is None:
        return None
    if mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
          mask_engine = None
    elif mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    options = ProcessOptions(get_processing_plugins(mask_engine), roop.globals.distance_threshold, roop.globals.blend_ratio,
                              "all", 0, clip_text, None, 0, 128, False, False, True)

    current_frame = live_swap(current_frame, options)
    return util.convert_to_gradio(current_frame)


def on_clear_input_faces():
    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    return ui.globals.ui_input_thumbs

def on_clear_destfiles():
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs.clear()
    return ui.globals.ui_target_thumbs


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
                selected_mask_engine, clip_text, processing_method, no_face_action, vr_mode, autorotate, restore_original_mouth, num_swap_steps, upsample, imagemask, progress=gr.Progress()):
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

    batch_process_regular(output_method, list_files_process, mask_engine, clip_text, processing_method == "In-Memory processing", imagemask, restore_original_mouth, num_swap_steps, progress, SELECTED_INPUT_FACE_INDEX)
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
    
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        if total_frames is None or total_frames < 1:
            total_frames = 1
            gr.Warning(f"Corrupted video {filename}, can't detect number of frames!")
        else:
            current_video_fps = util.detect_fps(filename)
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
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        current_video_fps = util.detect_fps(filename)
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
