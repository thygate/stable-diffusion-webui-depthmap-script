# non-public API (don't host publicly) 
# (will only be on with --api starting option)

import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from PIL import Image

import gradio as gr

from modules.api.models import *
from modules.api import api
from modules.shared import opts

from src.core import core_generation_funnel
from src.common_ui import main_ui_panel
from src.misc import SCRIPT_VERSION

def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return api.encode_pil_to_base64(image)
    elif type(image) is np.ndarray:
        return encode_np_to_base64(image)
    else:
        return ""

def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)

def to_base64_PIL(encoding: str):
    return Image.fromarray(np.array(api.decode_base64_to_image(encoding)).astype('uint8'))

#TODO: is this slow?
def get_defaults():
    default_gradio = main_ui_panel(True).internal
    defaults = {}
    for key, value in default_gradio.items(): 
        defaults[key]= value.value
    return defaults

def depth_api(_: gr.Blocks, app: FastAPI):
    @app.get("/depth/version")
    async def version():
        return {"version": SCRIPT_VERSION}

    @app.get("/depth/get_options")
    async def get_options():
        default_input =  get_defaults()
        return {"settings": sorted(list(default_input.internal.keys()))}
    
    #This will be the stable basic api
    @app.post("/depth/process")
    async def process(
        depth_input_images: List[str] = Body([], title='Input Images'),
        compute_device:str = Body("GPU", title='CPU or GPU', options="'GPU', 'CPU'"), 
        model_type:str = Body('zoedepth_n (indoor)', title='depth model', options="'res101', 'dpt_beit_large_512 (midas 3.1)', 'dpt_beit_large_384 (midas 3.1)', 'dpt_large_384 (midas 3.0)', 'dpt_hybrid_384 (midas 3.0)', 'midas_v21', 'midas_v21_small', 'zoedepth_n (indoor)', 'zoedepth_k (outdoor)', 'zoedepth_nk'"),
        net_width:int = Body(512, title="net width"), 
        net_height:int = Body(512, title="net height"), 
        match_size:bool =  Body(True, title="match original image size"), 
        boost:bool = Body(False, title="use boost algorithm"), 
        invert_depth:bool = Body(False, title="invert depthmap")
    ):
        default_inputs = get_defaults()
        override = {
            # TODO: These indexing aren't soo nice
            'compute_device': compute_device, 
            'model_type': ['res101', 'dpt_beit_large_512 (midas 3.1)',
              'dpt_beit_large_384 (midas 3.1)', 'dpt_large_384 (midas 3.0)',
              'dpt_hybrid_384 (midas 3.0)',
              'midas_v21', 'midas_v21_small',
              'zoedepth_n (indoor)', 'zoedepth_k (outdoor)', 'zoedepth_nk'].index(model_type),
            'net_width': net_width, 
            'net_height': net_height, 
            'match_size':  match_size, 
            'boost': boost, 
            'invert_depth': invert_depth,
        }

        for key, value in override.items():
            default_inputs[key] = value

        if len(depth_input_images) == 0:
            raise HTTPException(
                status_code=422, detail="No image selected")

        print(f"Processing {str(len(depth_input_images))} images with the depth module.")

        PIL_images = []
        for input_image in depth_input_images:
            PIL_images.append(to_base64_PIL(input_image))

        outpath = opts.outdir_samples or opts.outdir_extras_samples
        img_gen = core_generation_funnel(outpath, PIL_images, None, None, default_inputs)[0]

        # This just keeps depth image throws everything else away
        results = [img['depth'] for img in img_gen]
        results64 = list(map(encode_to_base64, results))

        return {"images": results64, "info": "Success"}
    
    #This will be direct process for overriding the default settings
    @app.post("/depth/raw_process")
    async def raw_process(
        depth_input_images: List[str] = Body([], title='Input Images'),
        override: dict = Body({}, title="a dictionary containing exact internal keys to depthmap")
    ):
        
        default_inputs = get_defaults()
        for key, value in override.items():
            default_inputs[key] = value

        if len(depth_input_images) == 0:
            raise HTTPException(
                status_code=422, detail="No image selected")

        print(f"Processing {str(len(depth_input_images))} images with the depth module.")

        PIL_images = []
        for input_image in depth_input_images:
            PIL_images.append(to_base64_PIL(input_image))

        outpath = opts.outdir_samples or opts.outdir_extras_samples
        img_gen = core_generation_funnel(outpath, PIL_images, None, None, default_inputs)[0]

        # This just keeps depth image throws everything else away
        results = [img['depth'] for img in img_gen]
        results64 = list(map(encode_to_base64, results))
        return {"images": results64, "info": "Success"}
    
    # TODO: add functionality
    # most different output formats (.obj, etc) should have different apis because otherwise network bloat might become a thing

    @app.post("/depth/extras_process")
    async def extras_process(
        depth_input_images: List[str] = Body([], title='Input Images')
    ):
        
        return {"images": depth_input_images, "info": "Success"}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(depth_api)
except:
    pass
