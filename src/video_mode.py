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
        try:
            import imageio_ffmpeg
            # Suppose there are in fact 16 bits per pixel
            # If this is not the case, this is not a 16-bit depthvideo, so no need to process it this way
            gen = imageio_ffmpeg.read_frames(path, pix_fmt='gray16le', bits_per_pixel=16)
            video_info = next(gen)
            if video_info['pix_fmt'] == 'gray16le':
                width, height = video_info['size']
                frames = []
                for frame in gen:
                    # Not sure if this is implemented somewhere else
                    result = np.frombuffer(frame, dtype='uint16')
                    result.shape = (height, width)  # Why does it work? I don't remotely have any idea.
                    frames += [Image.fromarray(result)]
                    # TODO: Wrapping frames into Pillow objects is wasteful
                return video_info['fps'], frames
        finally:
            if 'gen' in locals():
                gen.close()
    if suffix in ['.webm', '.mp4', '.avi']:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(path)
        frames = [Image.fromarray(x) for x in list(clip.iter_frames())]
        # TODO: Wrapping frames into Pillow objects is wasteful
        return clip.fps, frames
    else:
        try:
            return 1, [Image.open(path)]
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
        for v_format, codec in priority:
            try:
                br = f'{colorvids_bitrate}k' if codec not in ['png', 'rawvideo'] else None
                clip.write_videofile(os.path.join(path, f"{name}.{v_format}"), codec=codec, bitrate=br)
                done = True
                break
            except:
                traceback.print_exc()
        if not done:
            raise Exception('Saving the video failed!')


def process_predicitons(predictions, smoothening='none'):
    def global_scaling(objs, a=None, b=None):
        """Normalizes objs, but uses (a, b) instead of (minimum, maximum) value of objs, if supplied"""
        normalized = []
        min_value = a if a is not None else min([obj.min() for obj in objs])
        max_value = b if b is not None else max([obj.max() for obj in objs])
        for obj in objs:
            normalized += [(obj - min_value) / (max_value - min_value)]
        return normalized

    print('Processing generated depthmaps')
    # TODO: Detect cuts and process segments separately
    if smoothening == 'none':
        return global_scaling(predictions)
    elif smoothening == 'experimental':
        processed = []
        clip = lambda val: min(max(0, val), len(predictions) - 1)
        for i in range(len(predictions)):
            f = np.zeros_like(predictions[i])
            for u, mul in enumerate([0.10, 0.20, 0.40, 0.20, 0.10]):  # Eyeballed it, math person please fix this
                f += mul * predictions[clip(i + (u - 2))]
            processed += [f]
        # This could have been deterministic monte carlo... Oh well, this version is faster.
        a, b = np.percentile(np.stack(processed), [0.5, 99.5])
        return global_scaling(predictions, a, b)
    return predictions


def gen_video(video, outpath, inp, custom_depthmap=None, colorvids_bitrate=None, smoothening='none'):
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
        input_depths = [x[2] for x in list(gen_obj)]
        input_depths = process_predicitons(input_depths, smoothening)
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
        frames_to_video(fps, imgs, outpath, f"depthmap-{backbone.get_next_sequence_number(outpath, basename)}-{basename}",
                        colorvids_bitrate)
    print('All done. Video(s) saved!')
    return '<h3>Videos generated</h3>' if len(gens) > 1 else '<h3>Video generated</h3>' if len(gens) == 1 \
        else '<h3>Nothing generated - please check the settings and try again</h3>'
