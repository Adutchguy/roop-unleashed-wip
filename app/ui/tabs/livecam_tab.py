import gradio as gr
import roop.globals
import ui.globals


camera_frame = None

def livecam_tab():
    with gr.Tab("🎥 Live Cam"):
        with gr.Row(variant='panel'):
            gr.Markdown("""
                        This feature will allow you to use your physical webcam and apply the selected faces to the stream. 
                        You can also forward the stream to a virtual camera, which can be used in video calls or streaming software.<br />
                        Supported are: v4l2loopback (linux), OBS Virtual Camera (macOS/Windows) and unitycapture (Windows).<br />
                        **Please note:** to change the face or any other settings you need to stop and restart a running live cam.
            """)

        with gr.Row(variant='panel'):
            with gr.Column():
                bt_start = gr.Button("▶ Start", variant='primary')
            with gr.Column():
                bt_stop = gr.Button("⏹ Stop", variant='secondary', interactive=False)
            with gr.Column():
                camera_num = gr.Slider(0, 8, value=0, label="Camera Number", step=1.0, interactive=True)
                cb_obs = gr.Checkbox(label="Forward stream to virtual camera", interactive=True)
            with gr.Column():
                dd_reso = gr.Dropdown(choices=["640x480","1280x720", "1920x1080"], value="1280x720", label="Fake Camera Resolution", interactive=True)
                cb_xseg = gr.Checkbox(label="Use DFL Xseg masking", interactive=True, value=True)
                cb_mouthrestore = gr.Checkbox(label="Restore original mouth area", interactive=True, value=False)

        with gr.Row():
            fake_cam_image = gr.Image(label='Fake Camera Output', interactive=False, format="jpeg")

    start_event = bt_start.click(fn=start_cam,  inputs=[cb_obs, cb_xseg, cb_mouthrestore, camera_num, dd_reso, ui.globals.ui_selected_enhancer, ui.globals.ui_blend_ratio, ui.globals.ui_upscale],outputs=[bt_start, bt_stop,fake_cam_image])
    bt_stop.click(fn=stop_swap, cancels=[start_event], outputs=[bt_start, bt_stop], queue=False)


def start_cam(stream_to_obs, use_xseg, use_mouthrestore, cam, reso, enhancer, blend_ratio, upscale):
    from roop.virtualcam import start_virtual_cam
    from roop.utilities import convert_to_gradio

    roop.globals.selected_enhancer = enhancer
    roop.globals.blend_ratio = blend_ratio
    roop.globals.subsample_size = int(upscale[:3])
    start_virtual_cam(stream_to_obs, use_xseg, use_mouthrestore, cam, reso)
    while True:
        yield gr.Button(interactive=False), gr.Button(interactive=True), convert_to_gradio(ui.globals.ui_camera_frame)
        

def stop_swap():
    from roop.virtualcam import stop_virtual_cam
    stop_virtual_cam()
    return gr.Button(interactive=True), gr.Button(interactive=False)
    



