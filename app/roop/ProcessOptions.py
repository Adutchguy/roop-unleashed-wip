class ProcessOptions:

    def __init__(self, processordefines:dict, face_distance,  blend_ratio, swap_mode, selected_index, masking_text, imagemask, num_steps, subsample_size, show_face_area, restore_original_mouth, show_mask=False, use_3d_recon=False,
                 use_source_bank=False, use_frontalization=False, frontalization_threshold=25.0, swap_model='inswapper'):
        self.processors = processordefines
        self.face_distance_threshold = face_distance
        self.blend_ratio = blend_ratio
        self.swap_mode = swap_mode
        self.selected_index = selected_index
        self.masking_text = masking_text
        self.imagemask = imagemask
        self.num_swap_steps = num_steps
        self.show_face_area_overlay = show_face_area
        self.show_face_masking = show_mask
        self.subsample_size = subsample_size
        self.restore_original_mouth = restore_original_mouth
        self.max_num_reuse_frame = 15
        # 3D source pose matching
        self.use_3d_recon = use_3d_recon
        # Multi-angle source bank (Option 1)
        self.use_source_bank = use_source_bank
        # Target frontalization (Option 2)
        self.use_frontalization = use_frontalization
        self.frontalization_threshold = frontalization_threshold
        # Swap model selection (Option 4): 'inswapper' | 'ghost'
        self.swap_model = swap_model