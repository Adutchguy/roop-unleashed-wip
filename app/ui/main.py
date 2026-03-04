import os
import time
import warnings
import gradio as gr
import roop.globals
import roop.metadata
import roop.utilities as util
import ui.globals as uii
import ui.globals

from ui.tabs.faceswap_tab import faceswap_tab
from ui.tabs.livecam_tab import livecam_tab
from ui.tabs.facemgr_tab import facemgr_tab
from ui.tabs.extras_tab import extras_tab
from ui.tabs.settings_tab import settings_tab

roop.globals.keep_fps = None
roop.globals.keep_frames = None
roop.globals.skip_audio = None
roop.globals.use_batch = None

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def prepare_environment():
    roop.globals.output_path = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(roop.globals.output_path, exist_ok=True)
    if not roop.globals.CFG.use_os_temp_folder:
        os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(os.path.join(os.getcwd(), "temp"))
    os.makedirs(os.environ["TEMP"], exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = os.environ["TEMP"]
    os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'

def run():
    from roop.core import decode_execution_providers, set_display_ui

    prepare_environment()

    set_display_ui(show_msg)
    if roop.globals.CFG.provider == "cuda" and util.has_cuda_device() == False:
       roop.globals.CFG.provider = "cpu"

    roop.globals.execution_providers = decode_execution_providers([roop.globals.CFG.provider])
    gputype = util.get_device()
    if gputype == 'cuda':
        util.print_cuda_info()
        
    print(f'Using provider {roop.globals.execution_providers} - Device:{gputype}')
    
    run_server = True
    uii.ui_restart_server = False
    mycss = """
    /* ── Dark Forest Theme ───────────────────────────────────────────────────
       #000000  black       – page / body background
       #1a472a  dark-green  – block / panel backgrounds, tab bar, header
       #2a623d  mid-green   – secondary panels, inputs, hover surfaces
       #5d5d5d  gray        – borders, dividers, tracks
       #aaaaaa  light-gray  – muted / secondary text
       Text:    #ffffff on all dark surfaces
    ──────────────────────────────────────────────────────────────────────── */

    :root, .dark {
        color-scheme: dark !important;

        /* accent / links */
        --color-accent:                            #aaaaaa;
        --color-accent-soft:                       rgba(170,170,170,0.20);
        --border-color-accent:                     #aaaaaa;
        --border-color-primary:                    #5d5d5d;
        --link-text-color:                         #aaaaaa;
        --link-text-color-hover:                   #ffffff;
        --link-text-color-active:                  #ffffff;
        --link-text-color-visited:                 #aaaaaa;

        /* page backgrounds */
        --body-background-fill:                    #000000;
        --background-fill-primary:                 #1a472a;
        --background-fill-secondary:               #2a623d;

        /* blocks */
        --block-background-fill:                   #1a472a;
        --block-border-color:                      #5d5d5d;
        --block-border-width:                      1px;
        --block-label-background-fill:             #000000;
        --block-label-text-color:                  #aaaaaa;
        --block-title-text-color:                  #ffffff;
        --block-info-text-color:                   #aaaaaa;
        --block-radius:                            8px;

        /* panels */
        --panel-background-fill:                   #1a472a;
        --panel-border-color:                      #5d5d5d;

        /* inputs */
        --input-background-fill:                   #2a623d;
        --input-background-fill-focus:             #2a623d;
        --input-border-color:                      #5d5d5d;
        --input-border-color-focus:                #aaaaaa;
        --input-border-color-hover:                #aaaaaa;
        --input-shadow:                            none;
        --input-shadow-focus:                      0 0 0 3px rgba(170,170,170,0.25);
        --input-placeholder-color:                 #5d5d5d;
        --input-text-color:                        #ffffff;

        /* primary buttons */
        --button-primary-background-fill:          #2a623d;
        --button-primary-background-fill-hover:    #1a472a;
        --button-primary-text-color:               #ffffff;
        --button-primary-border-color:             #5d5d5d;
        --button-primary-border-color-hover:       #aaaaaa;
        --button-primary-shadow:                   none;
        --button-primary-shadow-hover:             0 2px 8px rgba(170,170,170,0.20);

        /* secondary buttons */
        --button-secondary-background-fill:        #1a472a;
        --button-secondary-background-fill-hover:  #2a623d;
        --button-secondary-text-color:             #aaaaaa;
        --button-secondary-border-color:           #5d5d5d;
        --button-secondary-border-color-hover:     #aaaaaa;

        /* stop/cancel buttons */
        --button-cancel-background-fill:           #5a1a1a;
        --button-cancel-background-fill-hover:     #7a2a2a;
        --button-cancel-text-color:                #ffffff;
        --button-cancel-border-color:              #5a1a1a;

        /* checkboxes – high-visibility on dark background */
        --checkbox-background-color:               #2a623d;
        --checkbox-background-color-focus:         #2a623d;
        --checkbox-background-color-selected:      #aaaaaa;
        --checkbox-background-color-hover:         #2a623d;
        --checkbox-border-color:                   #aaaaaa;
        --checkbox-border-color-focus:             #ffffff;
        --checkbox-border-color-selected:          #ffffff;
        --checkbox-border-color-hover:             #ffffff;
        --checkbox-label-background-fill:          transparent;
        --checkbox-label-background-fill-hover:    rgba(170,170,170,0.10);
        --checkbox-label-background-fill-selected: rgba(170,170,170,0.15);
        --checkbox-label-text-color:               #ffffff;

        /* slider */
        --slider-color:                            #aaaaaa;

        /* table */
        --table-odd-background-fill:               #1a472a;
        --table-even-background-fill:              #2a623d;
        --table-row-focus:                         rgba(170,170,170,0.10);

        /* shadows */
        --shadow-drop:                             0 1px 4px rgba(0,0,0,0.60);
        --shadow-drop-lg:                          0 4px 16px rgba(0,0,0,0.70);
        --shadow-inset:                            inset 0 1px 3px rgba(0,0,0,0.50);

        /* neutral scale */
        --neutral-50:  #ffffff;
        --neutral-100: #aaaaaa;
        --neutral-200: #5d5d5d;
        --neutral-300: #5d5d5d;
        --neutral-400: #5d5d5d;
        --neutral-500: #2a623d;
        --neutral-600: #1a472a;
        --neutral-700: #1a472a;
        --neutral-800: #000000;
        --neutral-900: #000000;
        --neutral-950: #000000;
    }

    /* ── Direct element overrides ── */

    html, body { background-color: #000000 !important; color: #ffffff !important; }
    .gradio-container, .gradio-container.dark {
        background: #000000 !important;
        color: #ffffff !important;
    }

    /* Blocks / panels */
    .block, .panel, fieldset, .form, .gap, .contain, .tabs {
        background: #1a472a !important;
        border: 1px solid #5d5d5d !important;
        color: #ffffff !important;
    }

    /* Block labels */
    .block-label, .block > .label-wrap, .block > label > span,
    label span, .block .label-wrap span {
        color: #aaaaaa !important;
    }

    /* General text */
    .block p, .block span, .block div { color: #ffffff !important; }

    /* Inputs (text, number, etc – checkboxes/radios handled separately) */
    input:not([type=range]):not([type=checkbox]):not([type=radio]), textarea, select {
        background: #2a623d !important;
        border: 1px solid #5d5d5d !important;
        color: #ffffff !important;
    }
    input:not([type=range]):not([type=checkbox]):not([type=radio]):focus, textarea:focus {
        border-color: #aaaaaa !important;
        box-shadow: 0 0 0 3px rgba(170,170,170,0.25) !important;
        background: #2a623d !important;
    }
    ::placeholder { color: #5d5d5d !important; opacity: 1; }

    /* Dropdowns / option lists */
    .wrap, ul.options, .dropdown-arrow {
        background: #2a623d !important;
        border: 1px solid #5d5d5d !important;
        color: #ffffff !important;
    }
    ul.options li { color: #ffffff !important; background: #2a623d !important; }
    ul.options li:hover, ul.options li.selected {
        background: #1a472a !important;
        color: #ffffff !important;
    }

    /* File upload / drop zones */
    .upload-container, .file-preview, .drop-container {
        background: #1a472a !important;
        border: 2px dashed #5d5d5d !important;
        color: #aaaaaa !important;
    }
    .drop-container:hover { border-color: #aaaaaa !important; }

    /* Gallery */
    .gallery, .gallery-container, .grid-container { background: #1a472a !important; }
    .gallery-item, .thumbnail-item { border: 1px solid #5d5d5d !important; }
    .gallery-item:hover, .thumbnail-item:hover {
        border-color: #aaaaaa !important;
        box-shadow: 0 0 0 2px rgba(170,170,170,0.30) !important;
    }

    /* Buttons – primary */
    button.primary, .btn-primary {
        background: #2a623d !important;
        border-color: #5d5d5d !important;
        color: #ffffff !important;
    }
    button.primary:hover, .btn-primary:hover {
        background: #1a472a !important;
        border-color: #aaaaaa !important;
    }

    /* Buttons – secondary */
    button.secondary, .btn-secondary {
        background: #1a472a !important;
        border-color: #5d5d5d !important;
        color: #aaaaaa !important;
    }
    button.secondary:hover, .btn-secondary:hover {
        background: #2a623d !important;
        border-color: #aaaaaa !important;
        color: #ffffff !important;
    }

    /* Buttons – stop / cancel */
    button.stop, button.cancel, .btn-cancel {
        background: #5a1a1a !important;
        border-color: #5a1a1a !important;
        color: #ffffff !important;
    }

    /* Sliders */
    input[type=range] { accent-color: #aaaaaa; }
    input[type=range]::-webkit-slider-thumb { background: #aaaaaa !important; }
    input[type=range]::-moz-range-thumb     { background: #aaaaaa !important; }
    input[type=range]::-webkit-slider-runnable-track { background: #1a472a !important; }

    /* Checkboxes – restore native rendering so checked state is always visible */
    input[type=checkbox] {
        appearance: auto !important;
        -webkit-appearance: checkbox !important;
        accent-color: #aaaaaa !important;
        width: 16px !important;
        height: 16px !important;
        cursor: pointer !important;
        background: unset !important;
        border: unset !important;
        box-shadow: none !important;
    }
    input[type=radio] { accent-color: #aaaaaa; }

    /* Header row */
    .compact {
        background: #000000 !important;
        border-bottom: 1px solid #5d5d5d !important;
    }

    /* Tab bar */
    .tab-nav { background: #000000 !important; border-bottom-color: #5d5d5d !important; }
    .tab-nav button {
        color: #aaaaaa !important;
        font-weight: 500;
        background: transparent !important;
    }
    .tab-nav button:hover {
        color: #ffffff !important;
        background: rgba(170,170,170,0.10) !important;
    }
    .tab-nav button.selected {
        color: #ffffff !important;
        border-bottom: 2px solid #aaaaaa !important;
        font-weight: 700 !important;
        background: rgba(170,170,170,0.15) !important;
    }

    /* Accordion headers */
    .label-wrap {
        background: #000000 !important;
        border: 1px solid #5d5d5d !important;
        border-radius: 6px !important;
    }
    .label-wrap:hover { background: #1a472a !important; }
    .label-wrap span  { color: #ffffff !important; }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #1a472a; }
    ::-webkit-scrollbar-thumb { background: #5d5d5d; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #aaaaaa; }

    /* Progress / generating */
    .progress-bar { background-color: #aaaaaa !important; }
    .generating    { border-color: #aaaaaa !important; }

    /* Toast notifications */
    .toast-wrap  { background: #1a472a !important; border-color: #5d5d5d !important; }
    .toast-title { color: #ffffff !important; }
    .toast-text  { color: #aaaaaa !important; }

    /* Markdown / prose */
    .prose, .prose p, .prose li { color: #ffffff !important; }
    .prose a { color: #aaaaaa !important; }
    .prose a:hover { color: #ffffff !important; }

    /* ── preserved rules ── */
    span {color: var(--block-info-text-color)}
    #fixedheight {
        max-height: 238.4px;
        overflow-y: auto !important;
    }
    .image-container.svelte-1l6wqyv {height: 100%}
    """

    while run_server:
        server_name = roop.globals.CFG.server_name
        if server_name is None or len(server_name) < 1:
            server_name = None
        server_port = roop.globals.CFG.server_port
        if server_port <= 0:
            server_port = None
        ssl_verify = True
        with gr.Blocks(title=f'{roop.metadata.name} {roop.metadata.version}', theme=roop.globals.CFG.selected_theme, css=mycss, delete_cache=(60, 86400)) as ui:
            with gr.Row(variant='compact'):
                    gr.HTML(util.create_version_html(), elem_id="versions")
                    bt_save_session = gr.Button("💾 Save Settings", size='sm', variant='primary', scale=0)
                    bt_load_session = gr.Button("📂 Load Settings", size='sm', scale=0)
            bt_destfiles = faceswap_tab()
            livecam_tab()
            facemgr_tab()
            extras_tab(bt_destfiles)
            settings_tab()
            # Wire Save/Load after all tabs so ui.globals component refs are populated
            _comps = _session_components()
            bt_save_session.click(fn=save_session, inputs=_comps, outputs=[])
            bt_load_session.click(fn=load_session, inputs=[], outputs=_comps)
        launch_browser = roop.globals.CFG.launch_browser

        uii.ui_restart_server = False
        try:
            ui.queue().launch(inbrowser=launch_browser, server_name=server_name, server_port=server_port, share=roop.globals.CFG.server_share, ssl_verify=ssl_verify, prevent_thread_lock=True, show_error=True)
        except Exception as e:
            print(f'Exception {e} when launching Gradio Server!')
            uii.ui_restart_server = True
            run_server = False
        try:
            while uii.ui_restart_server == False:
                time.sleep(1.0)

        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()


def show_msg(msg: str):
    gr.Info(msg)


_SESSION_CFG_KEYS = [
    'face_detection_mode', 'num_swap_steps', 'selected_enhancer', 'max_face_distance',
    'subsample_upscale', 'blend_ratio', 'video_swapping_method', 'no_face_action',
    'vr_mode', 'autorotate_faces', 'skip_audio', 'keep_frames', 'wait_after_extraction',
    'output_method', 'mask_engine', 'mask_clip_text', 'show_mask_offsets',
    'restore_original_mouth', 'mask_top', 'mask_bottom', 'mask_left', 'mask_right',
    'mask_erosion', 'mask_blur',
]


def _session_components():
    return [
        ui.globals.ui_selected_face_detection,
        ui.globals.ui_num_swap_steps,
        ui.globals.ui_selected_enhancer,
        ui.globals.ui_max_face_distance,
        ui.globals.ui_upscale,
        ui.globals.ui_blend_ratio,
        ui.globals.ui_video_swapping_method,
        ui.globals.ui_no_face_action,
        ui.globals.ui_vr_mode,
        ui.globals.ui_autorotate,
        ui.globals.ui_skip_audio,
        ui.globals.ui_keep_frames,
        ui.globals.ui_wait_after_extraction,
        ui.globals.ui_output_method,
        ui.globals.ui_selected_mask_engine,
        ui.globals.ui_clip_text,
        ui.globals.ui_chk_showmaskoffsets,
        ui.globals.ui_chk_restoreoriginalmouth,
        ui.globals.ui_mask_top,
        ui.globals.ui_mask_bottom,
        ui.globals.ui_mask_left,
        ui.globals.ui_mask_right,
        ui.globals.ui_mask_erosion,
        ui.globals.ui_mask_blur,
    ]


def save_session(*values):
    cfg = roop.globals.CFG
    for key, val in zip(_SESSION_CFG_KEYS, values):
        setattr(cfg, key, val)
    cfg.save()
    gr.Info('Settings saved!')


def load_session():
    roop.globals.CFG.load()
    cfg = roop.globals.CFG
    return tuple(getattr(cfg, key) for key in _SESSION_CFG_KEYS)

