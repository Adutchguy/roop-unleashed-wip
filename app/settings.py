import yaml

class Settings:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load()

    def default_get(_, data, name, default):
        value = default
        try:
            value = data.get(name, default)
        except:
            pass
        return value


    def load(self):
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except:
            data = None

        self.selected_theme = self.default_get(data, 'selected_theme', "Default")
        self.server_name = self.default_get(data, 'server_name', "")
        self.server_port = self.default_get(data, 'server_port', 0)
        self.server_share = self.default_get(data, 'server_share', False)
        self.output_image_format = self.default_get(data, 'output_image_format', 'png')
        self.output_video_format = self.default_get(data, 'output_video_format', 'mp4')
        self.output_video_codec = self.default_get(data, 'output_video_codec', 'libx264')
        self.video_quality = self.default_get(data, 'video_quality', 14)
        self.clear_output = self.default_get(data, 'clear_output', True)
        self.max_threads = self.default_get(data, 'max_threads', 2)
        self.memory_limit = self.default_get(data, 'memory_limit', 0)
        self.provider = self.default_get(data, 'provider', 'cuda')
        self.force_cpu = self.default_get(data, 'force_cpu', False)
        self.output_template = self.default_get(data, 'output_template', '{file}_{time}')
        self.use_os_temp_folder = self.default_get(data, 'use_os_temp_folder', False)
        self.output_show_video = self.default_get(data, 'output_show_video', True)
        self.launch_browser = self.default_get(data, 'launch_browser', False)
        self.max_face_distance = self.default_get(data, 'max_face_distance', 0.85)
        # Faceswap session settings
        self.face_detection_mode = self.default_get(data, 'face_detection_mode', 'All faces')
        self.num_swap_steps = self.default_get(data, 'num_swap_steps', 1)
        self.selected_enhancer = self.default_get(data, 'selected_enhancer', 'GPEN')
        self.subsample_upscale = self.default_get(data, 'subsample_upscale', '256px')
        self.blend_ratio = self.default_get(data, 'blend_ratio', 0.80)
        self.video_swapping_method = self.default_get(data, 'video_swapping_method', 'In-Memory processing')
        self.no_face_action = self.default_get(data, 'no_face_action', 'Retry rotated')
        self.vr_mode = self.default_get(data, 'vr_mode', False)
        self.autorotate_faces = self.default_get(data, 'autorotate_faces', True)
        self.skip_audio = self.default_get(data, 'skip_audio', False)
        self.keep_frames = self.default_get(data, 'keep_frames', False)
        self.wait_after_extraction = self.default_get(data, 'wait_after_extraction', False)
        self.output_method = self.default_get(data, 'output_method', 'File')
        self.mask_engine = self.default_get(data, 'mask_engine', 'DFL XSeg')
        self.mask_clip_text = self.default_get(data, 'mask_clip_text', 'cup,hands,hair,banana')
        self.show_mask_offsets = self.default_get(data, 'show_mask_offsets', False)
        self.restore_original_mouth = self.default_get(data, 'restore_original_mouth', False)
        self.mask_top = self.default_get(data, 'mask_top', 0.0)
        self.mask_bottom = self.default_get(data, 'mask_bottom', 0.0)
        self.mask_left = self.default_get(data, 'mask_left', 0.0)
        self.mask_right = self.default_get(data, 'mask_right', 0.0)
        self.face_mask_blend = self.default_get(data, 'face_mask_blend', 20.0)
        self.mouth_mask_blend = self.default_get(data, 'mouth_mask_blend', 10.0)





    def save(self):
        data = {
            'selected_theme': self.selected_theme,
            'server_name': self.server_name,
            'server_port': self.server_port,
            'server_share': self.server_share,
            'output_image_format' : self.output_image_format,
            'output_video_format' : self.output_video_format,
            'output_video_codec' : self.output_video_codec,
            'video_quality' : self.video_quality,
            'clear_output' : self.clear_output,
            'max_threads' : self.max_threads,
            'memory_limit' : self.memory_limit,
            'provider' : self.provider,
            'force_cpu' : self.force_cpu,
            'output_template' : self.output_template,
            'use_os_temp_folder' : self.use_os_temp_folder,
            'output_show_video' : self.output_show_video,
            'launch_browser': self.launch_browser,
            'max_face_distance': self.max_face_distance,
            # Faceswap session settings
            'face_detection_mode': self.face_detection_mode,
            'num_swap_steps': self.num_swap_steps,
            'selected_enhancer': self.selected_enhancer,
            'subsample_upscale': self.subsample_upscale,
            'blend_ratio': self.blend_ratio,
            'video_swapping_method': self.video_swapping_method,
            'no_face_action': self.no_face_action,
            'vr_mode': self.vr_mode,
            'autorotate_faces': self.autorotate_faces,
            'skip_audio': self.skip_audio,
            'keep_frames': self.keep_frames,
            'wait_after_extraction': self.wait_after_extraction,
            'output_method': self.output_method,
            'mask_engine': self.mask_engine,
            'mask_clip_text': self.mask_clip_text,
            'show_mask_offsets': self.show_mask_offsets,
            'restore_original_mouth': self.restore_original_mouth,
            'mask_top': self.mask_top,
            'mask_bottom': self.mask_bottom,
            'mask_left': self.mask_left,
            'mask_right': self.mask_right,
            'face_mask_blend': self.face_mask_blend,
            'mouth_mask_blend': self.mouth_mask_blend,
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(data, f)



