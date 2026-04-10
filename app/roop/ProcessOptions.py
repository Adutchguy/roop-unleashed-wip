class ProcessOptions:

    def __init__(self, processordefines:dict, face_distance,  blend_ratio, swap_mode, selected_index, masking_text, imagemask, num_steps, subsample_size, show_face_area, restore_original_mouth, show_mask=False, restore_occluders=False, occluder_blend=0.8, temporal_threshold=30.0):
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
        # Occluder restoration (#3 skin-tone layer, #5 temporal detection)
        self.restore_occluders = restore_occluders
        self.occluder_blend = occluder_blend          # 0.0–1.0 restore strength
        self.temporal_threshold = temporal_threshold  # pixel deviation to flag as artifact