
import os
import subprocess
import roop.globals
import roop.utilities as util

from typing import List, Any

def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-y', '-loglevel', roop.globals.log_level]
    commands.extend(args)
    print ("Running ffmpeg")
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print("Running ffmpeg failed! Commandline:")
        print (" ".join(commands))
    return False



def cut_video(original_video: str, cut_video: str, start_frame: int, end_frame: int, reencode: bool):
    fps = util.detect_fps(original_video)
    start_time = start_frame / fps
    num_frames = end_frame - start_frame

    if reencode:
        run_ffmpeg(['-ss',  format(start_time, ".2f"), '-i', original_video, '-c:v', roop.globals.video_encoder, '-c:a', 'aac', '-frames:v', str(num_frames), cut_video])
    else:
        run_ffmpeg(['-ss',  format(start_time, ".2f"), '-i', original_video,  '-frames:v', str(num_frames), '-c:v' ,'copy','-c:a' ,'copy', cut_video])

def join_videos(videos: List[str], dest_filename: str, simple: bool):
    if simple:
        txtfilename = util.resolve_relative_path('../temp')
        txtfilename = os.path.join(txtfilename, 'joinvids.txt')
        with open(txtfilename, "w", encoding="utf-8") as f:
            for v in videos:
                 v = v.replace('\\', '/')
                 f.write(f"file {v}\n")
        commands = ['-f', 'concat', '-safe', '0', '-i', f'{txtfilename}', '-vcodec', 'copy', f'{dest_filename}']
        run_ffmpeg(commands)

    else:
        inputs = []
        filter = ''
        for i,v in enumerate(videos):
            inputs.append('-i')
            inputs.append(v)
            filter += f'[{i}:v:0][{i}:a:0]'
        run_ffmpeg([" ".join(inputs), '-filter_complex', f'"{filter}concat=n={len(videos)}:v=1:a=1[outv][outa]"', '-map', '"[outv]"', '-map', '"[outa]"', dest_filename])    

        #     filter += f'[{i}:v:0][{i}:a:0]'
        # run_ffmpeg([" ".join(inputs), '-filter_complex', f'"{filter}concat=n={len(videos)}:v=1:a=1[outv][outa]"', '-map', '"[outv]"', '-map', '"[outa]"', dest_filename])    



def extract_frames(target_path : str, trim_frame_start, trim_frame_end, fps : float) -> bool:
    util.create_temp(target_path)
    temp_directory_path = util.get_temp_directory_path(target_path)
    commands = ['-i', target_path, '-q:v', '1', '-pix_fmt', 'rgb24', ]
    if trim_frame_start is not None and trim_frame_end is not None:
        commands.extend([ '-vf', 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(trim_frame_end) + ',fps=' + str(fps) ])
    commands.extend(['-vsync', '0', os.path.join(temp_directory_path, '%06d.' + roop.globals.CFG.output_image_format)])
    return run_ffmpeg(commands)


def create_video(target_path: str, dest_filename: str, fps: float = 24.0, temp_directory_path: str = None) -> None:
    if temp_directory_path is None:
        temp_directory_path = util.get_temp_directory_path(target_path)
    run_ffmpeg(['-r', str(fps), '-i', os.path.join(temp_directory_path, f'%06d.{roop.globals.CFG.output_image_format}'), '-c:v', roop.globals.video_encoder, '-crf', str(roop.globals.video_quality), '-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', dest_filename])
    return dest_filename


def create_gif_from_video(video_path: str, gif_path):
    fps = util.detect_fps(video_path)
    width, height = util.detect_dimensions(video_path)

    # Keep the larger dimension at its original size; auto-scale the other.
    if width >= height:
        scale = f'{width}:-1'
    else:
        scale = f'-1:{height}'

    run_ffmpeg(['-i', video_path, '-vf', f'fps={fps},scale={scale}:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse', '-loop', '0', gif_path])


def apply_media_transforms_gif(input_path: str, output_path: str,
                                vf_filters: list, target_fps=None) -> bool:
    """Re-encode an animated GIF with correct palette generation.

    FFmpeg's default GIF encoder uses a poor global palette that introduces
    colour artifacts on grayscale content.  This function uses the two-pass
    palettegen+paletteuse pipeline so the output palette is optimised for the
    actual frame content — exactly the same approach used by create_gif_from_video.

    vf_filters  - list of video filters to apply BEFORE palette generation
                  (e.g. crop, scale, transpose).  May be empty.
    target_fps  - if not None, a fps= filter is prepended so the frame-rate
                  of the output GIF matches the requested value.
    """
    all_filters = list(vf_filters)
    if target_fps is not None:
        all_filters.insert(0, f'fps={target_fps}')

    if all_filters:
        user_chain = ','.join(all_filters) + ','
    else:
        user_chain = ''

    # Two-pass palette approach in a single ffmpeg invocation using filtergraph.
    # The filter chain is: [user filters] → split → palettegen / paletteuse
    vf = f'{user_chain}split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse'
    return run_ffmpeg(['-i', input_path, '-vf', vf, '-loop', '0', output_path])



def create_video_from_gif(gif_path: str, output_path):
    fps = util.detect_fps(gif_path)
    filter = """scale='trunc(in_w/2)*2':'trunc(in_h/2)*2',format=yuv420p,fps=10"""
    run_ffmpeg(['-i', gif_path, '-vf', f'"{filter}"', '-movflags', '+faststart', '-shortest', output_path])



def resize_video(input_path: str, output_path: str, width: int, height: int) -> bool:
    scale_filter = (
        f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
        f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
    )
    return run_ffmpeg(['-i', input_path, '-vf', scale_filter,
                       '-c:v', roop.globals.video_encoder,
                       '-crf', str(roop.globals.video_quality),
                       '-c:a', 'copy', output_path])


def rotate_media(input_path: str, output_path: str, transform: str) -> bool:
    transform_map = {
        "90° Clockwise":        "transpose=1",
        "90° Counter-clockwise": "transpose=2",
        "180°":                  "transpose=1,transpose=1",
        "Flip Horizontal":       "hflip",
        "Flip Vertical":         "vflip",
    }
    vf = transform_map.get(transform, "transpose=1")
    return run_ffmpeg(['-i', input_path, '-vf', vf, '-c:a', 'copy', output_path])


def change_fps(input_path: str, output_path: str, fps: float) -> bool:
    return run_ffmpeg(['-i', input_path, '-vf', f'fps={fps}',
                       '-c:v', roop.globals.video_encoder,
                       '-crf', str(roop.globals.video_quality),
                       '-c:a', 'copy', output_path])


def crop_media(input_path: str, output_path: str,
               left_pct: float, right_pct: float,
               top_pct: float,  bottom_pct: float) -> bool:
    l, r, t, b = left_pct / 100, right_pct / 100, top_pct / 100, bottom_pct / 100
    crop_filter = (
        f"crop=in_w*(1-{l:.4f}-{r:.4f}):in_h*(1-{t:.4f}-{b:.4f})"
        f":in_w*{l:.4f}:in_h*{t:.4f}"
    )
    return run_ffmpeg(['-i', input_path, '-vf', crop_filter, '-c:a', 'copy', output_path])


def apply_media_transforms(input_path: str, output_path: str,
                           vf_filters: list, is_video: bool) -> bool:
    """Apply a list of -vf filters in a single ffmpeg pass."""
    if not vf_filters:
        return False
    codec   = roop.globals.video_encoder   or 'libx264'
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14
    vf = ','.join(vf_filters)
    args = ['-i', input_path, '-vf', vf]
    if is_video:
        args += ['-c:v', codec, '-crf', str(quality), '-c:a', 'copy']
    args.append(output_path)
    return run_ffmpeg(args)


def apply_media_transforms_webp(input_path: str, output_path: str,
                                vf_filters: list, fps: float) -> bool:
    """Process animated webp: decode frames via PIL, pipe through ffmpeg with vf filters.

    FFmpeg cannot reliably decode animated webp files with malformed Exif headers.
    This function bypasses that by loading frames with Pillow and feeding raw BGR
    video into ffmpeg via stdin, applying any vf filters in a single pass.
    Output is always an mp4 (caller must ensure output_path has .mp4 extension).
    """
    import subprocess
    import numpy as np
    import cv2
    from PIL import Image

    try:
        frames = []
        width = height = 0
        with Image.open(input_path) as img:
            width, height = img.width, img.height
            for i in range(getattr(img, 'n_frames', 1)):
                img.seek(i)
                frame_bgr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)
    except Exception as e:
        print(f"apply_media_transforms_webp: failed to load frames: {e}")
        return False

    if not frames or width == 0 or height == 0:
        print("apply_media_transforms_webp: no frames or zero dimensions")
        return False

    # video_encoder/quality may be None if faceswap tab hasn't run yet — use safe defaults
    codec   = roop.globals.video_encoder   or 'libx264'
    quality = roop.globals.video_quality   if roop.globals.video_quality is not None else 14

    vf = ','.join(vf_filters)
    cmd = [
        'ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-y',
        '-loglevel', roop.globals.log_level,
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-an', '-i', '-',
        '-vf', vf,
        '-c:v', codec,
        '-crf', str(quality),
        '-pix_fmt', 'yuv420p',
        output_path,
    ]
    print(f"apply_media_transforms_webp: piping {len(frames)} frames @ {fps} fps")
    print(' '.join(cmd))

    try:
        popen_params = {'stdin': subprocess.PIPE, 'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE}
        if os.name == 'nt':
            popen_params['creationflags'] = 0x08000000  # CREATE_NO_WINDOW
        proc = subprocess.Popen(cmd, **popen_params)
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"apply_media_transforms_webp ffmpeg error:\n{stderr.decode(errors='replace')}")
        return proc.returncode == 0
    except Exception as e:
        print(f"apply_media_transforms_webp: subprocess failed: {e}")
        return False


def restore_audio(intermediate_video: str, original_video: str, trim_frame_start, trim_frame_end, final_video : str) -> None:
	fps = util.detect_fps(original_video)
	commands = [ '-i', intermediate_video ]
	if trim_frame_start is None and trim_frame_end is None:
		commands.extend([ '-c:a', 'copy' ])
	else:
		# if trim_frame_start is not None:
		# 	start_time = trim_frame_start / fps
		# 	commands.extend([ '-ss', format(start_time, ".2f")])
		# else:
		# 	commands.extend([ '-ss', '0' ])
		# if trim_frame_end is not None:
		# 	end_time = trim_frame_end / fps
		# 	commands.extend([ '-to', format(end_time, ".2f")])
		# commands.extend([ '-c:a', 'aac' ])
		if trim_frame_start is not None:
			start_time = trim_frame_start / fps
			commands.extend([ '-ss', format(start_time, ".2f")])
		else:
			commands.extend([ '-ss', '0' ])
		if trim_frame_end is not None:
			end_time = trim_frame_end / fps
			commands.extend([ '-to', format(end_time, ".2f")])
		commands.extend([ '-i', original_video, "-c",  "copy" ])

	commands.extend([ '-map', '0:v:0', '-map', '1:a:0?', '-shortest', final_video ])
	run_ffmpeg(commands)
