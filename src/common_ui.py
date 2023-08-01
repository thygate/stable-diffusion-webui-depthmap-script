import traceback
from pathlib import Path
import gradio as gr
from PIL import Image

from src import backbone, video_mode
from src.core import core_generation_funnel, unload_models, run_makevideo
from src.depthmap_generation import ModelHolder
from src.gradio_args_transport import GradioComponentBundle
from src.misc import *
from src.common_constants import GenerationOptions as go

# Ugly workaround to fix gradio tempfile issue
def ensure_gradio_temp_directory():
    try:
        import tempfile
        path = os.path.join(tempfile.gettempdir(), 'gradio')
        if not (os.path.exists(path)):
            os.mkdir(path)
    except Exception as e:
        traceback.print_exc()


ensure_gradio_temp_directory()


def main_ui_panel(is_depth_tab):
    inp = GradioComponentBundle()
    # TODO: Greater visual separation
    with gr.Blocks():
        with gr.Row() as cur_option_root:
            inp -= 'depthmap_gen_row_0', cur_option_root
            inp += go.COMPUTE_DEVICE, gr.Radio(label="Compute on", choices=['GPU', 'CPU'], value='GPU')
            # TODO: Should return value instead of index. Maybe Enum should be used?
            inp += go.MODEL_TYPE, gr.Dropdown(label="Model",
                                             choices=['res101', 'dpt_beit_large_512 (midas 3.1)',
                                                      'dpt_beit_large_384 (midas 3.1)', 'dpt_large_384 (midas 3.0)',
                                                      'dpt_hybrid_384 (midas 3.0)',
                                                      'midas_v21', 'midas_v21_small',
                                                      'zoedepth_n (indoor)', 'zoedepth_k (outdoor)', 'zoedepth_nk'],
                                             type="index")
        with gr.Box() as cur_option_root:
            inp -= 'depthmap_gen_row_1', cur_option_root
            with gr.Row():
                inp += go.BOOST, gr.Checkbox(label="BOOST (multi-resolution merging)")
                inp += go.NET_SIZE_MATCH, gr.Checkbox(label="Match net size to input size", visible=False)
            with gr.Row(visible=False) as options_depend_on_match_size:
                inp += go.NET_WIDTH, gr.Slider(minimum=64, maximum=2048, step=64, label='Net width')
                inp += go.NET_HEIGHT, gr.Slider(minimum=64, maximum=2048, step=64, label='Net height')

        with gr.Box() as cur_option_root:
            inp -= 'depthmap_gen_row_2', cur_option_root
            with gr.Row():
                with gr.Group():  # 50% of width
                    inp += "save_outputs", gr.Checkbox(label="Save Outputs", value=True)
                with gr.Group():  # 50% of width
                    inp += go.DO_OUTPUT_DEPTH, gr.Checkbox(label="Output DepthMap")
                    inp += go.OUTPUT_DEPTH_INVERT, gr.Checkbox(label="Invert (black=near, white=far)")
            with gr.Row() as options_depend_on_output_depth_1:
                inp += go.OUTPUT_DEPTH_COMBINE, gr.Checkbox(
                    label="Combine input and depthmap into one image")
                inp += go.OUTPUT_DEPTH_COMBINE_AXIS, gr.Radio(
                    label="Combine axis", choices=['Vertical', 'Horizontal'], type="value", visible=False)

        with gr.Box() as cur_option_root:
            inp -= 'depthmap_gen_row_3', cur_option_root
            with gr.Row():
                inp += go.CLIPDEPTH, gr.Checkbox(label="Clip and renormalize DepthMap")
                inp += go.CLIPDEPTH_MODE,\
                    gr.Dropdown(label="Mode", choices=['Range', 'Outliers'], type="value", visible=False)
            with gr.Row(visible=False) as clip_options_row_1:
                inp += go.CLIPDEPTH_FAR, gr.Slider(minimum=0, maximum=1, step=0.001, label='Far clip')
                inp += go.CLIPDEPTH_NEAR, gr.Slider(minimum=0, maximum=1, step=0.001, label='Near clip')

        with gr.Box():
            with gr.Row():
                inp += go.GEN_STEREO, gr.Checkbox(label="Generate stereoscopic image(s)")
            with gr.Column(visible=False) as stereo_options:
                with gr.Row():
                    inp += go.STEREO_MODES, gr.CheckboxGroup(
                        ["left-right", "right-left", "top-bottom", "bottom-top", "red-cyan-anaglyph",
                         "left-only", "only-right", "cyan-red-reverseanaglyph"
                         ][0:8 if backbone.get_opt('depthmap_script_extra_stereomodes', False) else 5], label="Output")
                with gr.Row():
                    inp += go.STEREO_DIVERGENCE, gr.Slider(minimum=0.05, maximum=15.005, step=0.01,
                                                          label='Divergence (3D effect)')
                    inp += go.STEREO_SEPARATION, gr.Slider(minimum=-5.0, maximum=5.0, step=0.01,
                                                          label='Separation (moves images apart)')
                with gr.Row():
                    inp += go.STEREO_FILL_ALGO, gr.Dropdown(label="Gap fill technique",
                                                      choices=['none', 'naive', 'naive_interpolating', 'polylines_soft',
                                                               'polylines_sharp'],
                                                      type="value")
                    inp += go.STEREO_OFFSET_EXPONENT, gr.Slider(label="Magic exponent", minimum=1, maximum=2, step=1)
                    inp += go.STEREO_BALANCE, gr.Slider(minimum=-1.0, maximum=1.0, step=0.05,
                                                       label='Balance between eyes')

        with gr.Box():
            with gr.Row():
                inp += go.GEN_NORMALMAP, gr.Checkbox(label="Generate NormalMap")
            with gr.Column(visible=False) as normalmap_options:
                with gr.Row():
                    inp += go.NORMALMAP_PRE_BLUR, gr.Checkbox(label="Smooth before calculating normals")
                    inp += go.NORMALMAP_PRE_BLUR_KERNEL, gr.Slider(minimum=1, maximum=31, step=2, label='Pre-smooth kernel size', visible=False)
                    inp.add_rule(go.NORMALMAP_PRE_BLUR_KERNEL, 'visible-if', go.NORMALMAP_PRE_BLUR)
                with gr.Row():
                    inp += go.NORMALMAP_SOBEL, gr.Checkbox(label="Sobel gradient")
                    inp += go.NORMALMAP_SOBEL_KERNEL, gr.Slider(minimum=1, maximum=31, step=2, label='Sobel kernel size')
                    inp.add_rule(go.NORMALMAP_SOBEL_KERNEL, 'visible-if', go.NORMALMAP_SOBEL)
                with gr.Row():
                    inp += go.NORMALMAP_POST_BLUR, gr.Checkbox(label="Smooth after calculating normals")
                    inp += go.NORMALMAP_POST_BLUR_KERNEL, gr.Slider(minimum=1, maximum=31, step=2, label='Post-smooth kernel size', visible=False)
                    inp.add_rule(go.NORMALMAP_POST_BLUR_KERNEL, 'visible-if', go.NORMALMAP_POST_BLUR)
                with gr.Row():
                    inp += go.NORMALMAP_INVERT, gr.Checkbox(label="Invert")

        if backbone.get_opt('depthmap_script_gen_heatmap_from_ui', False):
            with gr.Box():
                with gr.Row():
                    inp += go.GEN_HEATMAP, gr.Checkbox(label="Generate HeatMap")

        with gr.Box():
            with gr.Column():
                inp += go.GEN_SIMPLE_MESH, gr.Checkbox(label="Generate simple 3D mesh")
            with gr.Column(visible=False) as mesh_options:
                with gr.Row():
                    gr.HTML(value="Generates fast, accurate only with ZoeDepth models and no boost, no custom maps.")
                with gr.Row():
                    inp += go.SIMPLE_MESH_OCCLUDE, gr.Checkbox(label="Remove occluded edges")
                    inp += go.SIMPLE_MESH_SPHERICAL, gr.Checkbox(label="Equirectangular projection")

        if is_depth_tab:
            with gr.Box():
                with gr.Column():
                    inp += go.GEN_INPAINTED_MESH, gr.Checkbox(
                        label="Generate 3D inpainted mesh")
                with gr.Column(visible=False) as inpaint_options_row_0:
                    gr.HTML("Generation is sloooow. Required for generating videos from mesh.")
                    inp += go.GEN_INPAINTED_MESH_DEMOS, gr.Checkbox(
                        label="Generate 4 demo videos with 3D inpainted mesh.")
                    gr.HTML("More options for generating video can be found in the Generate video tab.")

        with gr.Box():
            # TODO: it should be clear from the UI that there is an option of the background removal
            #  that does not use the model selected above
            with gr.Row():
                inp += go.GEN_REMBG, gr.Checkbox(label="Remove background")
            with gr.Column(visible=False) as bgrem_options:
                with gr.Row():
                    inp += go.SAVE_BACKGROUND_REMOVAL_MASKS, gr.Checkbox(label="Save the foreground masks")
                    inp += go.PRE_DEPTH_BACKGROUND_REMOVAL, gr.Checkbox(label="Pre-depth background removal")
                with gr.Row():
                    inp += go.REMBG_MODEL, gr.Dropdown(
                        label="Rembg Model", type="value",
                        choices=['u2net', 'u2netp', 'u2net_human_seg', 'silueta', "isnet-general-use", "isnet-anime"])

        with gr.Box():
            gr.HTML(f"{SCRIPT_FULL_NAME}<br/>")
            gr.HTML("Information, comment and share @ <a "
                    "href='https://github.com/thygate/stable-diffusion-webui-depthmap-script'>"
                    "https://github.com/thygate/stable-diffusion-webui-depthmap-script</a>")

        def update_default_net_size(model_type):
            w, h = ModelHolder.get_default_net_size(model_type)
            return inp[go.NET_WIDTH].update(value=w), inp[go.NET_HEIGHT].update(value=h)

        inp[go.MODEL_TYPE].change(
            fn=update_default_net_size,
            inputs=inp[go.MODEL_TYPE],
            outputs=[inp[go.NET_WIDTH], inp[go.NET_HEIGHT]]
        )

        inp[go.BOOST].change(  # Go boost! Wroom!..
            fn=lambda a, b: (inp[go.NET_SIZE_MATCH].update(visible=not a),
                             options_depend_on_match_size.update(visible=not a and not b)),
            inputs=[inp[go.BOOST], inp[go.NET_SIZE_MATCH]],
            outputs=[inp[go.NET_SIZE_MATCH], options_depend_on_match_size]
        )
        inp.add_rule(options_depend_on_match_size, 'visible-if-not', go.NET_SIZE_MATCH)

        inp.add_rule(options_depend_on_output_depth_1, 'visible-if', go.DO_OUTPUT_DEPTH)
        inp.add_rule(go.OUTPUT_DEPTH_INVERT, 'visible-if', go.DO_OUTPUT_DEPTH)
        inp.add_rule(go.OUTPUT_DEPTH_COMBINE_AXIS, 'visible-if', go.OUTPUT_DEPTH_COMBINE)
        inp.add_rule(go.CLIPDEPTH_MODE, 'visible-if', go.CLIPDEPTH)
        inp.add_rule(clip_options_row_1, 'visible-if', go.CLIPDEPTH)

        inp[go.CLIPDEPTH_FAR].change(
            fn=lambda a, b: a if b < a else b,
            inputs=[inp[go.CLIPDEPTH_FAR], inp[go.CLIPDEPTH_NEAR]],
            outputs=[inp[go.CLIPDEPTH_NEAR]],
            show_progress=False
        )
        inp[go.CLIPDEPTH_NEAR].change(
            fn=lambda a, b: a if b > a else b,
            inputs=[inp[go.CLIPDEPTH_NEAR], inp[go.CLIPDEPTH_FAR]],
            outputs=[inp[go.CLIPDEPTH_FAR]],
            show_progress=False
        )

        inp.add_rule(stereo_options, 'visible-if', go.GEN_STEREO)
        inp.add_rule(normalmap_options, 'visible-if', go.GEN_NORMALMAP)
        inp.add_rule(mesh_options, 'visible-if', go.GEN_SIMPLE_MESH)
        if is_depth_tab:
            inp.add_rule(inpaint_options_row_0, 'visible-if', go.GEN_INPAINTED_MESH)
        inp.add_rule(bgrem_options, 'visible-if', go.GEN_REMBG)

    return inp

def open_folder_action():
    # Adapted from stable-diffusion-webui
    f = backbone.get_outpath()
    if backbone.get_cmd_opt('hide_ui_dir_config', False):
        return
    if not os.path.exists(f) or not os.path.isdir(f):
        raise Exception("Couldn't open output folder")  # .isdir is security-related, do not remove!
    import platform
    import subprocess as sp
    path = os.path.normpath(f)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        sp.Popen(["open", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        sp.Popen(["wsl-open", path])
    else:
        sp.Popen(["xdg-open", path])


def depthmap_mode_video(inp):
    gr.HTML(value="Single video mode allows generating videos from videos. Please "
                  "keep in mind that all the frames of the video need to be processed - therefore it is important to "
                  "pick settings so that the generation is not too slow. For the best results, "
                  "use a zoedepth model, since they provide the highest level of coherency between frames.")
    inp += gr.File(elem_id='depthmap_vm_input', label="Video or animated file",
                   file_count="single", interactive=True, type="file")
    inp += gr.Checkbox(elem_id="depthmap_vm_custom_checkbox",
                       label="Use custom/pregenerated DepthMap video", value=False)
    inp += gr.Dropdown(elem_id="depthmap_vm_smoothening_mode", label="Smoothening",
                       type="value", choices=['none', 'experimental'], value='experimental')
    inp += gr.File(elem_id='depthmap_vm_custom', file_count="single",
                   interactive=True, type="file", visible=False)
    with gr.Row():
        inp += gr.Checkbox(elem_id='depthmap_vm_compress_checkbox', label="Compress colorvideos?", value=False)
        inp += gr.Slider(elem_id='depthmap_vm_compress_bitrate', label="Bitrate (kbit)", visible=False,
                         minimum=1000, value=15000, maximum=50000, step=250)

    inp.add_rule('depthmap_vm_custom', 'visible-if', 'depthmap_vm_custom_checkbox')
    inp.add_rule('depthmap_vm_smoothening_mode', 'visible-if-not', 'depthmap_vm_custom_checkbox')
    inp.add_rule('depthmap_vm_compress_bitrate', 'visible-if', 'depthmap_vm_compress_checkbox')

    return inp


custom_css = """
#depthmap_vm_input {height: 75px}
#depthmap_vm_custom {height: 75px}
"""


def on_ui_tabs():
    inp = GradioComponentBundle()
    with gr.Blocks(analytics_enabled=False, title="DepthMap", css=custom_css) as depthmap_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                inp += 'depthmap_mode', gr.HTML(visible=False, value='0')
                with gr.Tabs():
                    with gr.TabItem('Single Image') as depthmap_mode_0:
                        with gr.Group():
                            with gr.Row():
                                inp += gr.Image(label="Source", source="upload", interactive=True, type="pil",
                                                elem_id="depthmap_input_image")
                                # TODO: depthmap generation settings should disappear when using this
                                inp += gr.File(label="Custom DepthMap", file_count="single", interactive=True,
                                               type="file", elem_id='custom_depthmap_img', visible=False)
                        inp += gr.Checkbox(elem_id="custom_depthmap", label="Use custom DepthMap", value=False)
                    with gr.TabItem('Batch Process') as depthmap_mode_1:
                        inp += gr.File(elem_id='image_batch', label="Batch Process", file_count="multiple",
                                       interactive=True, type="file")
                    with gr.TabItem('Batch from Directory') as depthmap_mode_2:
                        inp += gr.Textbox(elem_id="depthmap_batch_input_dir", label="Input directory",
                                          **backbone.get_hide_dirs(),
                                          placeholder="A directory on the same machine where the server is running.")
                        inp += gr.Textbox(elem_id="depthmap_batch_output_dir", label="Output directory",
                                          **backbone.get_hide_dirs(),
                                          placeholder="Leave blank to save images to the default path.")
                        gr.HTML("Files in the output directory may be overwritten.")
                        inp += gr.Checkbox(elem_id="depthmap_batch_reuse",
                                           label="Skip generation and use (edited/custom) depthmaps "
                                                 "in output directory when a file already exists.",
                                           value=True)
                    with gr.TabItem('Single Video') as depthmap_mode_3:
                        inp = depthmap_mode_video(inp)
                submit = gr.Button('Generate', elem_id="depthmap_generate", variant='primary')
                inp |= main_ui_panel(True)  # Main panel is inserted here
                unloadmodels = gr.Button('Unload models', elem_id="depthmap_unloadmodels")

            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="mode_depthmap_output"):
                    with gr.TabItem('Depth Output'):
                        with gr.Group():
                            result_images = gr.Gallery(label='Output', show_label=False,
                                                       elem_id=f"depthmap_gallery").style(grid=4)
                        with gr.Column():
                            html_info = gr.HTML()
                        folder_symbol = '\U0001f4c2'  # 📂
                        gr.Button(folder_symbol, visible=not backbone.get_cmd_opt('hide_ui_dir_config', False)).click(
                            fn=lambda: open_folder_action(), inputs=[], outputs=[],
                        )

                    with gr.TabItem('3D Mesh'):
                        with gr.Group():
                            result_depthmesh = gr.Model3D(label="3d Mesh", clear_color=[1.0, 1.0, 1.0, 1.0])
                            with gr.Row():
                                # loadmesh = gr.Button('Load')
                                clearmesh = gr.Button('Clear')

                    with gr.TabItem('Generate video'):
                        # generate video
                        with gr.Group():
                            with gr.Row():
                                gr.Markdown("Generate video from inpainted(!) mesh.")
                            with gr.Row():
                                depth_vid = gr.Video(interactive=False)
                            with gr.Column():
                                vid_html_info_x = gr.HTML()
                                vid_html_info = gr.HTML()
                                fn_mesh = gr.Textbox(label="Input Mesh (.ply | .obj)", **backbone.get_hide_dirs(),
                                                     placeholder="A file on the same machine where "
                                                                 "the server is running.")
                            with gr.Row():
                                vid_numframes = gr.Textbox(label="Number of frames", value="300")
                                vid_fps = gr.Textbox(label="Framerate", value="40")
                                vid_format = gr.Dropdown(label="Format", choices=['mp4', 'webm'], value='mp4',
                                                         type="value", elem_id="video_format")
                                vid_ssaa = gr.Dropdown(label="SSAA", choices=['1', '2', '3', '4'], value='3',
                                                       type="value", elem_id="video_ssaa")
                            with gr.Row():
                                vid_traj = gr.Dropdown(label="Trajectory",
                                                       choices=['straight-line', 'double-straight-line', 'circle'],
                                                       value='double-straight-line', type="index",
                                                       elem_id="video_trajectory")
                                vid_shift = gr.Textbox(label="Translate: x, y, z", value="-0.015, 0.0, -0.05")
                                vid_border = gr.Textbox(label="Crop: top, left, bottom, right",
                                                        value="0.03, 0.03, 0.05, 0.03")
                                vid_dolly = gr.Checkbox(label="Dolly", value=False, elem_classes="smalltxt")
                            with gr.Row():
                                submit_vid = gr.Button('Generate Video', elem_id="depthmap_generatevideo",
                                                       variant='primary')

        inp += inp.enkey_tail()

        depthmap_mode_0.select(lambda: '0', None, inp['depthmap_mode'])
        depthmap_mode_1.select(lambda: '1', None, inp['depthmap_mode'])
        depthmap_mode_2.select(lambda: '2', None, inp['depthmap_mode'])
        depthmap_mode_3.select(lambda: '3', None, inp['depthmap_mode'])

        def custom_depthmap_change_fn(mode, zero_on, three_on):
            hide = mode == '0' and zero_on or mode == '3' and three_on
            return inp['custom_depthmap_img'].update(visible=hide), \
                inp['depthmap_gen_row_0'].update(visible=not hide), \
                inp['depthmap_gen_row_1'].update(visible=not hide), \
                inp['depthmap_gen_row_3'].update(visible=not hide), not hide
        custom_depthmap_change_els = ['depthmap_mode', 'custom_depthmap', 'depthmap_vm_custom_checkbox']
        for el in custom_depthmap_change_els:
            inp[el].change(
            fn=custom_depthmap_change_fn,
            inputs=[inp[el] for el in custom_depthmap_change_els],
            outputs=[inp[st] for st in [
                'custom_depthmap_img', 'depthmap_gen_row_0', 'depthmap_gen_row_1', 'depthmap_gen_row_3',
                go.DO_OUTPUT_DEPTH]])

        unloadmodels.click(
            fn=unload_models,
            inputs=[],
            outputs=[]
        )

        clearmesh.click(
            fn=lambda: None,
            inputs=[],
            outputs=[result_depthmesh]
        )

        submit.click(
            fn=backbone.wrap_gradio_gpu_call(run_generate),
            inputs=inp.enkey_body(),
            outputs=[
                result_images,
                fn_mesh,
                result_depthmesh,
                html_info
            ]
        )

        submit_vid.click(
            fn=backbone.wrap_gradio_gpu_call(run_makevideo),
            inputs=[
                fn_mesh,
                vid_numframes,
                vid_fps,
                vid_traj,
                vid_shift,
                vid_border,
                vid_dolly,
                vid_format,
                vid_ssaa
            ],
            outputs=[
                depth_vid,
                vid_html_info_x,
                vid_html_info
            ]
        )

    return depthmap_interface


def format_exception(e: Exception):
    traceback.print_exc()
    msg = '<h3>' + 'ERROR: ' + str(e) + '</h3>' + '\n'
    if 'out of GPU memory' not in msg:
        msg += \
            'Please report this issue ' \
            f'<a href="https://github.com/thygate/{REPOSITORY_NAME}/issues">here</a>. ' \
            'Make sure to provide the full stacktrace: \n'
        msg += '<code style="white-space: pre;">' + traceback.format_exc() + '</code>'
    return msg


def run_generate(*inputs):
    inputs = GradioComponentBundle.enkey_to_dict(inputs)
    depthmap_mode = inputs['depthmap_mode']
    depthmap_batch_input_dir = inputs['depthmap_batch_input_dir']
    image_batch = inputs['image_batch']
    depthmap_input_image = inputs['depthmap_input_image']
    depthmap_batch_output_dir = inputs['depthmap_batch_output_dir']
    depthmap_batch_reuse = inputs['depthmap_batch_reuse']
    custom_depthmap = inputs['custom_depthmap']
    custom_depthmap_img = inputs['custom_depthmap_img']

    inputimages = []
    inputdepthmaps = []  # Allow supplying custom depthmaps
    inputnames = []  # Also keep track of original file names

    if depthmap_mode == '3':
        try:
            custom_depthmap = inputs['depthmap_vm_custom'] \
                if inputs['depthmap_vm_custom_checkbox'] else None
            colorvids_bitrate = inputs['depthmap_vm_compress_bitrate'] \
                if inputs['depthmap_vm_compress_checkbox'] else None
            ret = video_mode.gen_video(
                inputs['depthmap_vm_input'], backbone.get_outpath(), inputs, custom_depthmap, colorvids_bitrate,
                inputs['depthmap_vm_smoothening_mode'])
            return [], None, None, ret
        except Exception as e:
            ret = format_exception(e)
        return [], None, None, ret

    if depthmap_mode == '2' and depthmap_batch_output_dir != '':
        outpath = depthmap_batch_output_dir
    else:
        outpath = backbone.get_outpath()

    if depthmap_mode == '0':  # Single image
        if depthmap_input_image is None:
            return [], None, None, "Please select an input image"
        inputimages.append(depthmap_input_image)
        inputnames.append(None)
        if custom_depthmap:
            if custom_depthmap_img is None:
                return [], None, None, \
                    "Custom depthmap is not specified. Please either supply it or disable this option."
            inputdepthmaps.append(Image.open(os.path.abspath(custom_depthmap_img.name)))
        else:
            inputdepthmaps.append(None)
    if depthmap_mode == '1':  # Batch Process
        if image_batch is None:
            return [], None, None, "Please select input images", ""
        for img in image_batch:
            image = Image.open(os.path.abspath(img.name))
            inputimages.append(image)
            inputnames.append(os.path.splitext(img.orig_name)[0])
        print(f'{len(inputimages)} images will be processed')
    elif depthmap_mode == '2':  # Batch from Directory
        # TODO: There is a RAM leak when we process batches, I can smell it! Or maybe it is gone.
        assert not backbone.get_cmd_opt('hide_ui_dir_config', False), '--hide-ui-dir-config option must be disabled'
        if depthmap_batch_input_dir == '':
            return [], None, None, "Please select an input directory."
        if depthmap_batch_input_dir == depthmap_batch_output_dir:
            return [], None, None, "Please pick different directories for batch processing."
        image_list = backbone.listfiles(depthmap_batch_input_dir)
        for path in image_list:
            try:
                inputimages.append(Image.open(path))
                inputnames.append(path)

                custom_depthmap = None
                if depthmap_batch_reuse:
                    basename = Path(path).stem
                    # Custom names are not used in samples directory
                    if outpath != backbone.get_opt('outdir_extras_samples', None):
                        # Possible filenames that the custom depthmaps may have
                        name_candidates = [f'{basename}-0000.{backbone.get_opt("samples_format", "png")}',  # current format
                                           f'{basename}.png',  # human-intuitive format
                                           f'{Path(path).name}']  # human-intuitive format (worse)
                        for fn_cand in name_candidates:
                            path_cand = os.path.join(outpath, fn_cand)
                            if os.path.isfile(path_cand):
                                custom_depthmap = Image.open(os.path.abspath(path_cand))
                                break
                inputdepthmaps.append(custom_depthmap)
            except Exception as e:
                print(f'Failed to load {path}, ignoring. Exception: {str(e)}')
        inputdepthmaps_n = len([1 for x in inputdepthmaps if x is not None])
        print(f'{len(inputimages)} images will be processed, {inputdepthmaps_n} existing depthmaps will be reused')

    gen_obj = core_generation_funnel(outpath, inputimages, inputdepthmaps, inputnames, inputs, backbone.gather_ops())

    # Saving images
    img_results = []
    results_total = 0
    inpainted_mesh_fi = mesh_simple_fi = None
    msg = ""  # Empty string is never returned
    while True:
        try:
            input_i, type, result = next(gen_obj)
            results_total += 1
        except StopIteration:
            # TODO: return more info
            msg = '<h3>Successfully generated</h3>' if results_total > 0 else \
                '<h3>Successfully generated nothing - please check the settings and try again</h3>'
            break
        except Exception as e:
            msg = format_exception(e)
            break
        if type == 'simple_mesh':
            mesh_simple_fi = result
            continue
        if type == 'inpainted_mesh':
            inpainted_mesh_fi = result
            continue
        if not isinstance(result, Image.Image):
            print(f'This is not supposed to happen! Somehow output type {type} is not supported! Input_i: {input_i}.')
            continue
        img_results += [(input_i, type, result)]

        if inputs["save_outputs"]:
            try:
                basename = 'depthmap'
                if depthmap_mode == '2' and inputnames[input_i] is not None:
                    if outpath != backbone.get_opt('outdir_extras_samples', None):
                        basename = Path(inputnames[input_i]).stem
                suffix = "" if type == "depth" else f"{type}"
                backbone.save_image(result, path=outpath, basename=basename, seed=None,
                           prompt=None, extension=backbone.get_opt('samples_format', 'png'), short_filename=True,
                           no_prompt=True, grid=False, pnginfo_section_name="extras",
                           suffix=suffix)
            except Exception as e:
                if not ('image has wrong mode' in str(e) or 'I;16' in str(e)):
                    raise e
                print('Catched exception: image has wrong mode!')
                traceback.print_exc()

    # Deciding what mesh to display (and if)
    display_mesh_fi = None
    if backbone.get_opt('depthmap_script_show_3d', True):
        display_mesh_fi = mesh_simple_fi
        if backbone.get_opt('depthmap_script_show_3d_inpaint', True):
            if inpainted_mesh_fi is not None and len(inpainted_mesh_fi) > 0:
                display_mesh_fi = inpainted_mesh_fi
    return map(lambda x: x[2], img_results), inpainted_mesh_fi, display_mesh_fi, msg.replace('\n', '<br>')
