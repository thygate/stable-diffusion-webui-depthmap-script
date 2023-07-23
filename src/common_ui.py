import traceback
from pathlib import Path
import gradio as gr
from PIL import Image

from src import backbone
from src.core import core_generation_funnel, unload_models, run_makevideo
from src.depthmap_generation import ModelHolder
from src.gradio_args_transport import GradioComponentBundle
from src.misc import *


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
        with gr.Row():
            inp += 'compute_device', gr.Radio(label="Compute on", choices=['GPU', 'CPU'], value='GPU')
            # TODO: Should return value instead of index. Maybe Enum should be used?
            inp += 'model_type', gr.Dropdown(label="Model",
                                             choices=['res101', 'dpt_beit_large_512 (midas 3.1)',
                                                      'dpt_beit_large_384 (midas 3.1)', 'dpt_large_384 (midas 3.0)',
                                                      'dpt_hybrid_384 (midas 3.0)',
                                                      'midas_v21', 'midas_v21_small',
                                                      'zoedepth_n (indoor)', 'zoedepth_k (outdoor)', 'zoedepth_nk'],
                                             value='res101',
                                             type="index")
        with gr.Box():
            with gr.Row():
                inp += 'boost', gr.Checkbox(label="BOOST (multi-resolution merging)", value=True)
                inp += 'match_size', gr.Checkbox(label="Match net size to input size", value=False)
            with gr.Row(visible=False) as options_depend_on_match_size:
                inp += 'net_width', gr.Slider(minimum=64, maximum=2048, step=64, label='Net width', value=448)
                inp += 'net_height', gr.Slider(minimum=64, maximum=2048, step=64, label='Net height', value=448)

        with gr.Box():
            with gr.Row():
                with gr.Group():
                    inp += "save_outputs", gr.Checkbox(label="Save Outputs", value=True)  # 50% of width
                with gr.Group():  # 50% of width
                    inp += "output_depth", gr.Checkbox(label="Output DepthMap", value=True)
                    inp += "invert_depth", gr.Checkbox(label="Invert (black=near, white=far)", value=False)
            with gr.Row() as options_depend_on_output_depth_1:
                inp += "combine_output", gr.Checkbox(
                    label="Combine input and depthmap into one image", value=False)
                inp += "combine_output_axis", gr.Radio(label="Combine axis", choices=['Vertical', 'Horizontal'],
                                                       value='Horizontal', type="index", visible=False)
        with gr.Box():
            with gr.Row():
                inp += 'clipdepth', gr.Checkbox(label="Clip and renormalize DepthMap", value=False)
            with gr.Row(visible=False) as clip_options_row_1:
                inp += "clipthreshold_far", gr.Slider(minimum=0, maximum=1, step=0.001, label='Far clip', value=0)
                inp += "clipthreshold_near", gr.Slider(minimum=0, maximum=1, step=0.001, label='Near clip', value=1)

        with gr.Box():
            with gr.Row():
                inp += "gen_stereo", gr.Checkbox(label="Generate stereoscopic image(s)", value=False)
            with gr.Column(visible=False) as stereo_options:
                with gr.Row():
                    inp += "stereo_modes", gr.CheckboxGroup(
                        ["left-right", "right-left", "top-bottom", "bottom-top", "red-cyan-anaglyph"],
                        label="Output", value=["left-right", "red-cyan-anaglyph"])
                with gr.Row():
                    inp += "stereo_divergence", gr.Slider(minimum=0.05, maximum=10.005, step=0.01,
                                                          label='Divergence (3D effect)',
                                                          value=2.5)
                    inp += "stereo_separation", gr.Slider(minimum=-5.0, maximum=5.0, step=0.01,
                                                          label='Separation (moves images apart)',
                                                          value=0.0)
                with gr.Row():
                    inp += "stereo_fill", gr.Dropdown(label="Gap fill technique",
                                                      choices=['none', 'naive', 'naive_interpolating', 'polylines_soft',
                                                               'polylines_sharp'], value='polylines_sharp',
                                                      type="value")
                    inp += "stereo_balance", gr.Slider(minimum=-1.0, maximum=1.0, step=0.05,
                                                       label='Balance between eyes',
                                                       value=0.0)

        with gr.Box():
            with gr.Row():
                inp += "gen_normalmap", gr.Checkbox(label="Generate NormalMap", value=False)
            with gr.Column(visible=False) as normalmap_options:
                with gr.Row():
                    inp += 'normalmap_pre_blur', gr.Checkbox(label="Smooth before calculating normals", value=False)
                    inp += 'normalmap_pre_blur_kernel', gr.Slider(minimum=1, maximum=31, step=2, label='Pre-smooth kernel size', value=3)
                with gr.Row():
                    inp += 'normalmap_sobel', gr.Checkbox(label="Sobel gradient", value=True)
                    inp += 'normalmap_sobel_kernel', gr.Slider(minimum=1, maximum=31, step=2, label='Sobel kernel size', value=3)
                with gr.Row():
                    inp += 'normalmap_post_blur', gr.Checkbox(label="Smooth after calculating normals", value=False)
                    inp += 'normalmap_post_blur_kernel', gr.Slider(minimum=1, maximum=31, step=2, label='Post-smooth kernel size', value=3)
                with gr.Row():
                    inp += 'normalmap_invert', gr.Checkbox(label="Invert", value=False)

        with gr.Box():
            with gr.Row():
                inp += "show_heat", gr.Checkbox(label="Generate HeatMap", value=False)

        with gr.Box():
            with gr.Column():
                inp += "gen_mesh", gr.Checkbox(
                    label="Generate simple 3D mesh", value=False, visible=True)
            with gr.Column(visible=False) as mesh_options:
                with gr.Row():
                    gr.HTML(value="Generates fast, accurate only with ZoeDepth models and no boost, no custom maps")
                with gr.Row():
                    inp += "mesh_occlude", gr.Checkbox(label="Remove occluded edges", value=True, visible=True)
                    inp += "mesh_spherical", gr.Checkbox(label="Equirectangular projection", value=False, visible=True)

        if is_depth_tab:
            with gr.Box():
                with gr.Column():
                    inp += "inpaint", gr.Checkbox(
                        label="Generate 3D inpainted mesh", value=False)
                with gr.Column(visible=False) as inpaint_options_row_0:
                    gr.HTML("Generation is sloooow, required for generating videos")
                    inp += "inpaint_vids", gr.Checkbox(
                        label="Generate 4 demo videos with 3D inpainted mesh.", value=False)
                    gr.HTML("More options for generating video can be found in the Generate video tab")

        with gr.Box():
            # TODO: it should be clear from the UI that there is an option of the background removal
            #  that does not use the model selected above
            with gr.Row():
                inp += "background_removal", gr.Checkbox(label="Remove background", value=False)
            with gr.Column(visible=False) as bgrem_options:
                with gr.Row():
                    inp += "save_background_removal_masks", gr.Checkbox(label="Save the foreground masks", value=False)
                    inp += "pre_depth_background_removal", gr.Checkbox(label="Pre-depth background removal", value=False)
                with gr.Row():
                    inp += "background_removal_model", gr.Dropdown(label="Rembg Model",
                                                                   choices=['u2net', 'u2netp', 'u2net_human_seg',
                                                                            'silueta'],
                                                                   value='u2net', type="value")

        with gr.Box():
            gr.HTML(f"{SCRIPT_FULL_NAME}<br/>")
            gr.HTML("Information, comment and share @ <a "
                    "href='https://github.com/thygate/stable-diffusion-webui-depthmap-script'>"
                    "https://github.com/thygate/stable-diffusion-webui-depthmap-script</a>")

        inp += "gen_normal", gr.Checkbox(label="Generate Normalmap (hidden! api only)", value=False, visible=False)

        def update_delault_net_size(model_type):
            w, h = ModelHolder.get_default_net_size(model_type)
            return inp['net_width'].update(value=w), inp['net_height'].update(value=h)

        inp['model_type'].change(
            fn=update_delault_net_size,
            inputs=inp['model_type'],
            outputs=[inp['net_width'], inp['net_height']]
        )

        inp['boost'].change(
            fn=lambda a, b: (inp['match_size'].update(visible=not a),
                             options_depend_on_match_size.update(visible=not a and not b)),
            inputs=[inp['boost'], inp['match_size']],
            outputs=[inp['match_size'], options_depend_on_match_size]
        )
        inp['match_size'].change(
            fn=lambda a, b: options_depend_on_match_size.update(visible=not a and not b),
            inputs=[inp['boost'], inp['match_size']],
            outputs=[options_depend_on_match_size]
        )

        inp['output_depth'].change(
            fn=lambda a: (inp['invert_depth'].update(visible=a), options_depend_on_output_depth_1.update(visible=a)),
            inputs=[inp['output_depth']],
            outputs=[inp['invert_depth'], options_depend_on_output_depth_1]
        )

        inp['combine_output'].change(
            fn=lambda v: inp['combine_output_axis'].update(visible=v),
            inputs=[inp['combine_output']],
            outputs=[inp['combine_output_axis']]
        )

        inp['clipdepth'].change(
            fn=lambda v: clip_options_row_1.update(visible=v),
            inputs=[inp['clipdepth']],
            outputs=[clip_options_row_1]
        )
        inp['clipthreshold_far'].change(
            fn=lambda a, b: a if b < a else b,
            inputs=[inp['clipthreshold_far'], inp['clipthreshold_near']],
            outputs=[inp['clipthreshold_near']]
        )
        inp['clipthreshold_near'].change(
            fn=lambda a, b: a if b > a else b,
            inputs=[inp['clipthreshold_near'], inp['clipthreshold_far']],
            outputs=[inp['clipthreshold_far']]
        )

        inp['gen_stereo'].change(
            fn=lambda v: stereo_options.update(visible=v),
            inputs=[inp['gen_stereo']],
            outputs=[stereo_options]
        )

        inp['gen_normalmap'].change(
            fn=lambda v: normalmap_options.update(visible=v),
            inputs=[inp['gen_normalmap']],
            outputs=[normalmap_options]
        )

        inp['gen_mesh'].change(
            fn=lambda v: mesh_options.update(visible=v),
            inputs=[inp['gen_mesh']],
            outputs=[mesh_options]
        )

        if is_depth_tab:
            inp['inpaint'].change(
                fn=lambda v: inpaint_options_row_0.update(visible=v),
                inputs=[inp['inpaint']],
                outputs=[inpaint_options_row_0]
            )

        inp['background_removal'].change(
            fn=lambda v: bgrem_options.update(visible=v),
            inputs=[inp['background_removal']],
            outputs=[bgrem_options]
        )

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

def on_ui_tabs():
    inp = GradioComponentBundle()
    with gr.Blocks(analytics_enabled=False, title="DepthMap") as depthmap_interface:
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
                submit = gr.Button('Generate', elem_id="depthmap_generate", variant='primary')
                inp += main_ui_panel(True)  # Main panel is inserted here
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

        inp['custom_depthmap'].change(
            fn=lambda v: inp['custom_depthmap_img'].update(visible=v),
            inputs=[inp['custom_depthmap']],
            outputs=[inp['custom_depthmap_img']]
        )

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
    # Allow supplying custom depthmaps
    inputdepthmaps = []
    # Also keep track of original file names
    inputnames = []

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
    elif depthmap_mode == '2':  # Batch from Directory
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

    outputs, fn_mesh, display_mesh = core_generation_funnel(outpath, inputimages, inputdepthmaps, inputnames, inputs, backbone.gather_ops())

    # Saving images
    show_images = []
    for input_i, imgs in enumerate(outputs):
        basename = 'depthmap'
        if depthmap_mode == '2' and inputnames[input_i] is not None and outpath != backbone.get_opt('outdir_extras_samples', None):
            basename = Path(inputnames[input_i]).stem

        for image_type, image in list(imgs.items()):
            show_images += [image]
            if inputs["save_outputs"]:
                try:
                    suffix = "" if image_type == "depth" else f"{image_type}"
                    backbone.save_image(image, path=outpath, basename=basename, seed=None,
                               prompt=None, extension=backbone.get_opt('samples_format', 'png'), short_filename=True,
                               no_prompt=True, grid=False, pnginfo_section_name="extras",
                               suffix=suffix)
                except Exception as e:
                    if not ('image has wrong mode' in str(e) or 'I;16' in str(e)):
                        raise e
                    print('Catched exception: image has wrong mode!')
                    traceback.print_exc()

    display_mesh = None
    # use inpainted 3d mesh to show in 3d model output when enabled in settings
    if backbone.get_opt('depthmap_script_show_3d_inpaint', True) and fn_mesh is not None and len(fn_mesh) > 0:
        display_mesh = fn_mesh
    # however, don't show 3dmodel when disabled in settings
    if not backbone.get_opt('depthmap_script_show_3d', True):
        display_mesh = None
    # TODO: return more info
    return show_images, fn_mesh, display_mesh, 'Generated!'
