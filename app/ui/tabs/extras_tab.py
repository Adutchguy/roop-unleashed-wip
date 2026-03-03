import os
import gradio as gr
import shutil
import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import roop.globals

frame_filters_map = { 
    "Colorize B/W Images (Deoldify Artistic)" : {"colorizer" : {"subtype": "deoldify_artistic"}},
    "Colorize B/W Images (Deoldify Stable)" : {"colorizer" : {"subtype": "deoldify_stable"}},
    "Background remove" : {"removebg" : {"subtype": ""}},
    "Filter Stylize" : {"filter_generic" : {"subtype" : "stylize" }},
    "Filter Detail Enhance" : {"filter_generic" : {"subtype" : "detailenhance" }},
    "Filter Pencil Sketch" : {"filter_generic" : {"subtype" : "pencil" }},
    "Filter Cartoon" : {"filter_generic" : {"subtype" : "cartoon" }},
    "Filter C64" : {"filter_generic" : {"subtype" : "C64" }}
    }

frame_upscalers_map = {
    "ESRGAN x2" : {"upscale" : {"subtype": "esrganx2"}},
    "ESRGAN x4" : {"upscale" : {"subtype": "esrganx4"}},
    "LSDIR x4" : {"upscale" : {"subtype": "lsdirx4"}}
}

def extras_tab():
    filternames = ["None"]
    for f in frame_filters_map.keys():
        filternames.append(f)
    upscalernames = ["None"]
    for f in frame_upscalers_map.keys():
        upscalernames.append(f)

    with gr.Tab("🎉 Extras"):
        with gr.Row():
            files_to_process = gr.Files(label='File(s) to process', file_count="multiple", file_types=["image", "video"])
        with gr.Row(variant='panel'):
            with gr.Accordion(label="Video/GIF", open=False):
                with gr.Row(variant='panel'):
                    with gr.Column():
                        gr.Markdown("""
                                    # Poor man's video editor
                                    Re-encoding uses your configuration from the Settings Tab.
    """)
                    with gr.Column():
                        cut_start_time = gr.Slider(0, 1000000, value=0, label="Start Frame", step=1.0, interactive=True)
                    with gr.Column():
                        cut_end_time = gr.Slider(1, 1000000, value=1, label="End Frame", step=1.0, interactive=True)
                    with gr.Column():
                        extras_chk_encode = gr.Checkbox(label='Re-encode videos (necessary for videos with different codecs)', value=False)
                        start_cut_video = gr.Button("Cut video")
                        start_extract_frames = gr.Button("Extract frames")
                        start_join_videos = gr.Button("Join videos")

                with gr.Row(variant='panel'):
                    with gr.Column():
                        gr.Markdown("""
                                    # Create video/gif from images
    """)
                    with gr.Column():
                        extras_fps = gr.Slider(minimum=0, maximum=120, value=30, label="Video FPS", step=1.0, interactive=True)
                        extras_images_folder = gr.Textbox(show_label=False, placeholder="/content/", interactive=True)
                    with gr.Column():
                        extras_chk_creategif = gr.Checkbox(label='Create GIF from video', value=False)
                        extras_create_video=gr.Button("Create")
                with gr.Row(variant='panel'):
                    with gr.Column():
                        gr.Markdown("""
                                    # Create video from gif
    """)
                    with gr.Column():
                        extras_video_fps = gr.Slider(minimum=0, maximum=120, value=0, label="Video FPS", step=1.0, interactive=True)
                    with gr.Column():
                        extras_create_video_from_gif=gr.Button("Create")

        with gr.Row(variant='panel'):
            with gr.Accordion(label="Full frame processing", open=True):
                with gr.Row(variant='panel'):
                    filterselection = gr.Dropdown(filternames, value="None", label="Colorizer/FilterFX", interactive=True)
                    upscalerselection = gr.Dropdown(upscalernames, value="None", label="Enhancer", interactive=True)
                with gr.Row(variant='panel'):
                    start_frame_process=gr.Button("Start processing")

        with gr.Row():
            gr.Button("👀 Open Output Folder", size='sm').click(fn=lambda: util.open_folder(roop.globals.output_path))
        with gr.Row():
            extra_files_output = gr.Files(label='Resulting output files', file_count="multiple")

    start_cut_video.click(fn=on_cut_video, inputs=[files_to_process, cut_start_time, cut_end_time, extras_chk_encode], outputs=[extra_files_output])
    start_extract_frames.click(fn=on_extras_extract_frames, inputs=[files_to_process], outputs=[extra_files_output])
    start_join_videos.click(fn=on_join_videos, inputs=[files_to_process, extras_chk_encode], outputs=[extra_files_output])
    extras_create_video.click(fn=on_extras_create_video, inputs=[files_to_process, extras_images_folder, extras_fps, extras_chk_creategif], outputs=[extra_files_output])
    extras_create_video_from_gif.click(fn=on_extras_create_video_from_gif, inputs=[files_to_process, extras_video_fps], outputs=[extra_files_output])
    start_frame_process.click(fn=on_frame_process, inputs=[files_to_process, filterselection, upscalerselection], outputs=[extra_files_output])


def on_cut_video(files, cut_start_frame, cut_end_frame, reencode):
    if files is None:
        return None
    
    resultfiles = []
    for tf in files:
        f = tf.name
        destfile = util.get_destfilename_from_path(f, roop.globals.output_path, '_cut')
        ffmpeg.cut_video(f, destfile, cut_start_frame, cut_end_frame, reencode)
        if os.path.isfile(destfile):
            resultfiles.append(destfile)
        else:
            gr.Error('Cutting video failed!')
    return resultfiles


def on_join_videos(files, chk_encode):
    if files is None:
        return None
    
    filenames = []
    for f in files:
        filenames.append(f.name)
    destfile = util.get_destfilename_from_path(filenames[0], roop.globals.output_path, '_join')
    sorted_filenames = util.sort_filenames_ignore_path(filenames)        
    ffmpeg.join_videos(sorted_filenames, destfile, not chk_encode)
    resultfiles = []
    if os.path.isfile(destfile):
        resultfiles.append(destfile)
    else:
        gr.Error('Joining videos failed!')
    return resultfiles

def on_extras_create_video_from_gif(files,fps):
    if files is None:
        return None
    
    filenames = []
    resultfiles = []
    for f in files:
        filenames.append(f.name)

    destfilename = os.path.join(roop.globals.output_path, "img2video." + roop.globals.CFG.output_video_format)
    ffmpeg.create_video_from_gif(filenames[0], destfilename)
    if os.path.isfile(destfilename):
        resultfiles.append(destfilename)
    return resultfiles





def on_extras_create_video(files, images_path,fps, create_gif):
    if images_path is None:
        return None
    resultfiles = []
    if len(files) > 0 and util.is_video(files[0]) and create_gif:
        destfilename = files[0]
    else:                     
        util.sort_rename_frames(os.path.dirname(images_path))
        destfilename = os.path.join(roop.globals.output_path, "img2video." + roop.globals.CFG.output_video_format)
        ffmpeg.create_video('', destfilename, fps, images_path)
        if os.path.isfile(destfilename):
            resultfiles.append(destfilename)
        else:
            return None
    if create_gif:
        gifname = util.get_destfilename_from_path(destfilename, './output', '.gif')
        ffmpeg.create_gif_from_video(destfilename, gifname)
        if os.path.isfile(destfilename):
            resultfiles.append(gifname)
    return resultfiles
    

def on_extras_extract_frames(files):
    if files is None:
        return None
    
    resultfiles = []
    for tf in files:
        f = tf.name
        resfolder = ffmpeg.extract_frames(f)
        for file in os.listdir(resfolder):
            outfile = os.path.join(resfolder, file)
            if os.path.isfile(outfile):
                resultfiles.append(outfile)
    return resultfiles


def on_frame_process(files, filterselection, upscaleselection):
    import pathlib
    from roop.core import batch_process_with_options
    from roop.ProcessEntry import ProcessEntry
    from roop.ProcessOptions import ProcessOptions
    from ui.main import prepare_environment


    if files is None:
        return None

    if roop.globals.CFG.clear_output:
        shutil.rmtree(roop.globals.output_path)
    prepare_environment()
    list_files_process : list[ProcessEntry] = []

    for tf in files:
        list_files_process.append(ProcessEntry(tf.name, 0,0, 0))

    processoroptions = {}
    filter = next((x for x in frame_filters_map.keys() if x == filterselection), None)
    if filter is not None:
        processoroptions.update(frame_filters_map[filter])
    filter = next((x for x in frame_upscalers_map.keys() if x == upscaleselection), None)
    if filter is not None:
        processoroptions.update(frame_upscalers_map[filter])
    options = ProcessOptions(processoroptions, 0,  0, "all", 0, None, None, 0, 128, False, False)
    batch_process_with_options(list_files_process, options, None)
    outdir = pathlib.Path(roop.globals.output_path)
    outfiles = [str(item) for item in outdir.rglob("*") if item.is_file()]
    return outfiles


