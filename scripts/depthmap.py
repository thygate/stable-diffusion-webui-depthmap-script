import traceback
import gradio as gr
from modules import shared
import modules.scripts as scripts
from PIL import Image

from src import backbone
from src import common_ui
from src.core import core_generation_funnel
from src.gradio_args_transport import GradioComponentBundle
from src.misc import *


class Script(scripts.Script):
    def title(self):
        return SCRIPT_NAME

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.HTML()  # Work around a Gradio bug
        with gr.Column(variant='panel'):
            gr.HTML()  # Work around a Gradio bug
            ret = common_ui.main_ui_panel(False)
            ret += ret.enkey_tail()
        return ret.enkey_body()

    # run from script in txt2img or img2img
    def run(self, p, *inputs):
        from modules import processing
        from modules.processing import create_infotext

        inputs = GradioComponentBundle.enkey_to_dict(inputs)

        # sd process
        processed = processing.process_images(p)
        processed.sampler = p.sampler  # for create_infotext

        inputimages = []
        for count in range(0, len(processed.images)):
            # skip first grid image
            if count == 0 and len(processed.images) > 1 and shared.opts.return_grid:
                continue
            inputimages.append(processed.images[count])

        gen_obj = core_generation_funnel(p.outpath_samples, inputimages, None, None, inputs, backbone.gather_ops())

        for input_i, type, result in gen_obj:
            if not isinstance(result, Image.Image):
                continue

            # get generation parameters
            # TODO: could reuse
            if hasattr(processed, 'all_prompts') and shared.opts.enable_pnginfo:
                info = create_infotext(
                    processed, processed.all_prompts, processed.all_seeds, processed.all_subseeds, "", 0, input_i)
            else:
                info = None

            processed.images.append(result)
            if inputs["save_outputs"]:
                try:
                    suffix = "" if type == "depth" else f"{type}"
                    backbone.save_image(result, path=p.outpath_samples, basename="", seed=processed.all_seeds[input_i],
                               prompt=processed.all_prompts[input_i], extension=shared.opts.samples_format,
                               info=info,
                               p=processed,
                               suffix=suffix)
                except Exception as e:
                    if not ('image has wrong mode' in str(e) or 'I;16' in str(e)):
                        raise e
                    print('Catched exception: image has wrong mode!')
                    traceback.print_exc()
        return processed


# TODO: some of them may be put into the main ui pane
# TODO: allow in standalone mode
def on_ui_settings():
    section = ('depthmap-script', "Depthmap extension")

    def add_option(name, default_value, description, name_prefix='depthmap_script'):
        shared.opts.add_option(f"{name_prefix}_{name}", shared.OptionInfo(default_value, description, section=section))

    add_option('keepmodels', False, "Do not unload depth and pix2pix models.")
    add_option('boost_rmax', 1600, "Maximum wholesize for boost (Rmax)")
    add_option('save_ply', False, "Save additional PLY file with 3D inpainted mesh.")
    add_option('show_3d', True, "Enable showing 3D Meshes in output tab. (Experimental)")
    add_option('show_3d_inpaint', True, "Also show 3D Inpainted Mesh in 3D Mesh output tab. (Experimental)")
    add_option('mesh_maxsize', 2048, "Max size for generating simple mesh.")
    add_option('gen_heatmap_from_ui', False, "Show an option to generate HeatMap in the UI")
    add_option('extra_stereomodes', False, "Enable more possible outputs for stereoimage generation")


from modules import script_callbacks
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(lambda: [(common_ui.on_ui_tabs(), "Depth", "depthmap_interface")])
