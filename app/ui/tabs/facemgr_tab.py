import os
import shutil
import cv2
import gradio as gr
import roop.utilities as util
import roop.globals
from roop.face_util import extract_face_images
from roop.capturer import get_video_frame, get_video_frame_total
from typing import List, Tuple, Optional
from roop.typing import Frame, Face, FaceSet

selected_face_index = -1
thumbs = []
images = []


def facemgr_tab() -> None:
    with gr.Tab("👨‍👩‍👧‍👦 Face Management"):
        with gr.Row():
            gr.Markdown("""
                        # Create blending facesets
                        Add multiple reference images into a faceset file.
                        """)
        with gr.Row():
            videoimagefst = gr.Image(label="Cut face from video frame", height=576, interactive=False, visible=True, format="jpeg")
        with gr.Row():
            frame_num_fst = gr.Slider(1, 1, value=1, label="Frame Number", info='0:00:00', step=1.0, interactive=False)
            fb_cutfromframe = gr.Button("Use faces from this frame", variant='secondary', interactive=False)
        with gr.Row():
            fb_facesetfile = gr.Files(label='Faceset', file_count='single', file_types=['.fsz'], interactive=True)
            fb_files = gr.Files(label='Input Files', file_count="multiple", file_types=["image", "video"], interactive=True)
        with gr.Row():
            with gr.Column():
                gr.Button("👀 Open Output Folder", size='sm').click(fn=lambda: util.open_folder(roop.globals.output_path))
            with gr.Column():
                gr.Markdown(' ')
        with gr.Row():
            faces = gr.Gallery(label="Faces in this Faceset", allow_preview=True, preview=True, height=128, object_fit="scale-down")
        with gr.Row():
            fb_remove = gr.Button("Remove selected", variant='secondary')
            fb_update = gr.Button("Create/Update Faceset file", variant='primary')
            fb_clear = gr.Button("Clear all", variant='stop')

    fb_facesetfile.change(fn=on_faceset_changed, inputs=[fb_facesetfile], outputs=[faces])
    fb_files.change(fn=on_fb_files_changed, inputs=[fb_files], outputs=[faces, videoimagefst, frame_num_fst, fb_cutfromframe])
    fb_update.click(fn=on_update_clicked, outputs=[fb_facesetfile])
    fb_remove.click(fn=on_remove_clicked, outputs=[faces])
    fb_clear.click(fn=on_clear_clicked, outputs=[faces, fb_files, fb_facesetfile])
    fb_cutfromframe.click(fn=on_cutfromframe_clicked, inputs=[fb_files, frame_num_fst], outputs=[faces])
    frame_num_fst.release(fn=on_frame_num_fst_changed, inputs=[fb_files, frame_num_fst], outputs=[videoimagefst])
    faces.select(fn=on_face_selected)


def on_faceset_changed(faceset, progress=gr.Progress()) -> List[Frame]:
    global thumbs, images

    if faceset is None:
        return thumbs

    thumbs.clear()
    filename = faceset.name
        
    if filename.lower().endswith('fsz'):
        progress(0, desc="Retrieving faces from Faceset File", )      
        unzipfolder = os.path.join(os.environ["TEMP"], 'faceset')
        if os.path.isdir(unzipfolder):
            shutil.rmtree(unzipfolder)
        util.mkdir_with_umask(unzipfolder)
        util.unzip(filename, unzipfolder)
        for file in os.listdir(unzipfolder):
            if file.endswith(".png"):
                SELECTION_FACES_DATA = extract_face_images(os.path.join(unzipfolder,file),  (False, 0), 0.5)
                if len(SELECTION_FACES_DATA) < 1:
                    gr.Warning(f"No face detected in {file}!")
                for f in SELECTION_FACES_DATA:
                    image = f[1]
                    images.append(image)
                    thumbs.append(util.convert_to_gradio(image))
        
        return thumbs


def on_fb_files_changed(inputfiles, progress=gr.Progress()) -> Tuple[List[Frame], Optional[gr.Image], Optional[gr.Slider], Optional[gr.Button]]:
    global thumbs, images, total_frames, current_video_fps

    if inputfiles is None or len(inputfiles) < 1:
        return thumbs, None, None, None
    
    progress(0, desc="Retrieving faces from images", )
    slider = None
    video_image = None
    cut_button = None
    for f in inputfiles:
        source_path = f.name
        if util.has_image_extension(source_path):
            slider = gr.Slider(interactive=False)
            video_image = gr.Image(interactive=False)
            cut_button = gr.Button(interactive=False)
            roop.globals.source_path = source_path
            SELECTION_FACES_DATA = extract_face_images(roop.globals.source_path,  (False, 0), 0.5)
            for f in SELECTION_FACES_DATA:
                image = f[1]
                images.append(image)
                thumbs.append(util.convert_to_gradio(image))
        elif util.is_video(source_path) or source_path.lower().endswith('gif'):
            total_frames = get_video_frame_total(source_path)
            current_video_fps = util.detect_fps(source_path)
            cut_button = gr.Button(interactive=True)
            video_image, slider = display_video_frame(source_path, 1, total_frames)

    return thumbs, video_image, slider, cut_button
    

def display_video_frame(filename: str, frame_num: int, total: int=0) -> Tuple[gr.Image, gr.Slider]:
    global current_video_fps

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
    if total > 0:
        return gr.Image(value=util.convert_to_gradio(current_frame), interactive=True), gr.Slider(info=timeinfo, minimum=1, maximum=total, interactive=True)  
    return gr.Image(value=util.convert_to_gradio(current_frame), interactive=True), gr.Slider(info=timeinfo, interactive=True)  


def on_face_selected(evt: gr.SelectData) -> None:
    global selected_face_index

    if evt is not None:
        selected_face_index = evt.index

def on_frame_num_fst_changed(inputfiles: List[gr.Files], frame_num: int) -> Frame:
    filename = inputfiles[0].name
    video_image, _ = display_video_frame(filename, frame_num, 0)
    return video_image


def on_cutfromframe_clicked(inputfiles: List[gr.Files], frame_num: int) -> List[Frame]:
    global thumbs

    filename = inputfiles[0].name
    SELECTION_FACES_DATA = extract_face_images(filename,  (True, frame_num), 0.5)
    for f in SELECTION_FACES_DATA:
        image = f[1]
        images.append(image)
        thumbs.append(util.convert_to_gradio(image))
    return thumbs


def on_remove_clicked() -> List[Frame]:
    global thumbs, images, selected_face_index

    if len(thumbs) > selected_face_index:
        f = thumbs.pop(selected_face_index)
        del f
        f = images.pop(selected_face_index)
        del f
    return thumbs

def on_clear_clicked() -> Tuple[List[Frame], None, None]:
    global thumbs, images

    thumbs.clear()
    images.clear()
    return thumbs, None, None


def on_update_clicked() -> Optional[str]:
    if len(images) < 1:
        gr.Warning(f"No faces to create faceset from!")
        return None

    imgnames = []
    for index,img in enumerate(images):
        filename = os.path.join(roop.globals.output_path, f'{index}.png')
        cv2.imwrite(filename, img)
        imgnames.append(filename)

    finalzip = os.path.join(roop.globals.output_path, 'faceset.fsz')        
    util.zip(imgnames, finalzip)
    return finalzip
