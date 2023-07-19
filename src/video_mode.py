import pathlib
import traceback

from PIL import Image
import numpy as np
import os

from src import core
from src import backbone
from src.common_constants import GenerationOptions as go


def open_path_as_images(path, maybe_depthvideo=False):
    """Takes the filepath, returns (fps, frames). Every frame is a Pillow Image object"""
    suffix = pathlib.Path(path).suffix
    if suffix == '.gif':
        frames = []
        img = Image.open(path)
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(img.convert('RGB'))
        return 1000 / img.info['duration'], frames
    if suffix in ['.avi'] and maybe_depthvideo:
        import imageio_ffmpeg
        gen = imageio_ffmpeg.read_frames(path)
        try:
            video_info = next(gen)
            if video_info['pix_fmt'] == 'gray16le':
                width, height = video_info['size']
                frames = []
                for frame in gen:
                    # Not sure if this is implemented somewhere else
                    result = np.frombuffer(frame, dtype='uint16')
                    result.shape = (height, width * 3 // 2)  # Why does it work? I don't remotely have any idea.
                    frames += [Image.fromarray(result)]
                    # TODO: Wrapping frames into Pillow objects is wasteful
                return video_info['fps'], frames
        finally:
            gen.close()
    if suffix in ['.webm', '.mp4', '.avi']:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(path)
        frames = [Image.fromarray(x) for x in list(clip.iter_frames())]
        # TODO: Wrapping frames into Pillow objects is wasteful
        return clip.fps, frames
    else:
        try:
            return 1000, [Image.open(path)]
        except Exception as e:
            raise Exception(f"Probably an unsupported file format: {suffix}") from e


def frames_to_video(fps, frames, path, name, colorvids_bitrate=None):
    if frames[0].mode == 'I;16':  # depthmap video
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(
            os.path.join(path, f"{name}.avi"), frames[0].size, 'gray16le', 'gray16le', fps, codec='ffv1',
            macro_block_size=1)
        try:
            writer.send(None)
            for frame in frames:
                writer.send(np.array(frame))
        finally:
            writer.close()
    else:
        arrs = [np.asarray(frame) for frame in frames]
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip(arrs, fps=fps)
        done = False
        priority = [('avi', 'png'), ('avi', 'rawvideo'), ('mp4', 'libx264'), ('webm', 'libvpx')]
        if colorvids_bitrate:
            priority = reversed(priority)
        for format, codec in priority:
            try:
                br = f'{colorvids_bitrate}k' if codec not in ['png', 'rawvideo'] else None
                clip.write_videofile(os.path.join(path, f"{name}.{format}"), codec=codec, bitrate=br)
                done = True
                break
            except:
                traceback.print_exc()
        if not done:
            raise Exception('Saving the video failed!')


def process_predicitons(predictions):
    print('Processing generated depthmaps')
    # TODO: Smart normalizing (drop 0.001% of top and bottom values from the video/every cut)
    preds_min_value = min([pred.min() for pred in predictions])
    preds_max_value = max([pred.max() for pred in predictions])

    input_depths = []
    for pred in predictions:
        norm = (pred - preds_min_value) / (preds_max_value - preds_min_value)  # normalize to [0; 1]
        input_depths += [norm]
    # TODO: Smoothening between frames (use splines)
    # TODO: Detect cuts and process segments separately

    return input_depths


def gen_video(video, outpath, inp, custom_depthmap=None, colorvids_bitrate=None):
    if inp[go.GEN_SIMPLE_MESH.name.lower()] or inp[go.GEN_INPAINTED_MESH.name.lower()]:
        return 'Creating mesh-videos is not supported. Please split video into frames and use batch processing.'

    fps, input_images = open_path_as_images(os.path.abspath(video.name))
    os.makedirs(backbone.get_outpath(), exist_ok=True)

    if custom_depthmap is None:
        print('Generating depthmaps for the video frames')
        needed_keys = [go.COMPUTE_DEVICE, go.MODEL_TYPE, go.BOOST, go.NET_SIZE_MATCH, go.NET_WIDTH, go.NET_HEIGHT]
        needed_keys = [x.name.lower() for x in needed_keys]
        first_pass_inp = {k: v for (k, v) in inp.items() if k in needed_keys}
        # We need predictions where frames are not normalized separately.
        first_pass_inp[go.DO_OUTPUT_DEPTH_PREDICTION] = True
        # No need in normalized frames. Properly processed depth video will be created in the second pass
        first_pass_inp[go.DO_OUTPUT_DEPTH.name] = False

        gen_obj = core.core_generation_funnel(None, input_images, None, None, first_pass_inp)
        predictions = [x[2] for x in list(gen_obj)]
        input_depths = process_predicitons(predictions)
    else:
        print('Using custom depthmap video')
        cdm_fps, input_depths = open_path_as_images(os.path.abspath(custom_depthmap.name), maybe_depthvideo=True)
        assert len(input_depths) == len(input_images), 'Custom depthmap video length does not match input video length'
        if input_depths[0].size != input_images[0].size:
            print('Warning! Input video size and depthmap video size are not the same!')

    print('Generating output frames')
    img_results = list(core.core_generation_funnel(None, input_images, input_depths, None, inp))
    gens = list(set(map(lambda x: x[1], img_results)))

    print('Saving generated frames as video outputs')
    for gen in gens:
        if gen == 'depth' and custom_depthmap is not None:
            # Well, that would be extra stupid, even if user has picked this option for some reason
            # (forgot to change the default?)
            continue

        imgs = [x[2] for x in img_results if x[1] == gen]
        basename = f'{gen}_video'
        frames_to_video(fps, imgs, outpath, f"depthmap-{backbone.get_next_sequence_number()}-{basename}",
                        colorvids_bitrate)
    print('All done. Video(s) saved!')
    return 'Video generated!' if len(gens) == 1 else 'Videos generated!'
