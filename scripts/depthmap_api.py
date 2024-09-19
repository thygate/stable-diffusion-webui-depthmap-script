# DO NOT HOST PUBLICLY - SECURITY RISKS!
# (the API will only be on with --api starting option)
# Currently no API stability guarantees are provided - API may break on any new commit (but hopefully won't).

import os
import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from PIL import Image

import gradio as gr

from typing import Dict, List
from modules.api import api

from src.core import core_generation_funnel, run_makevideo
from src.misc import SCRIPT_VERSION
from src import backbone
from src.common_constants import GenerationOptions as go


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


def depth_api(_: gr.Blocks, app: FastAPI):
    @app.get("/depth/version")
    async def version():
        return {"version": SCRIPT_VERSION}

    @app.get("/depth/get_options")
    async def get_options():
        return {"options": sorted([x.name.lower() for x in go])}

    # TODO: some potential inputs not supported (like custom depthmaps)
    @app.post("/depth/generate")
    async def process(
        depth_input_images: List[str] = Body([], title='Input Images'),
        options: Dict[str, object] = Body("options", title='Generation options'),
    ):
        # TODO: restrict mesh options

        if len(depth_input_images) == 0:
            raise HTTPException(status_code=422, detail="No images supplied")
        print(f"Processing {str(len(depth_input_images))} images through the API")

        pil_images = []
        for input_image in depth_input_images:
            pil_images.append(to_base64_PIL(input_image))
        outpath = backbone.get_outpath()
        gen_obj = core_generation_funnel(outpath, pil_images, None, None, options)

        results_based = []
        for count, type, result in gen_obj:
            if not isinstance(result, Image.Image):
                continue
            results_based += [encode_to_base64(result)]

        return {"images": results_based, "info": "Success"}

    @app.post("/depth/generate/video")
    async def process_video(
        depth_input_images: List[str] = Body([], title='Input Images'),
        options: Dict[str, object] = Body("options", title='Generation options'),
    ):
        if len(depth_input_images) == 0:
            raise HTTPException(status_code=422, detail="No images supplied")
        print(f"Processing {str(len(depth_input_images))} images through the API")

        # You can use either these strings, or integers
        available_models = {
            'res101': 0,
            'dpt_beit_large_512': 1, #midas 3.1
            'dpt_beit_large_384': 2, #midas 3.1
            'dpt_large_384': 3, #midas 3.0
            'dpt_hybrid_384': 4, #midas 3.0
            'midas_v21': 5,
            'midas_v21_small': 6,
            'zoedepth_n': 7, #indoor
            'zoedepth_k': 8, #outdoor
            'zoedepth_nk': 9,
            'marigold_v1': 10,
            'depth_anything': 11,
            'depth_anything_v2_small': 12,
            'depth_anything_v2_base': 13,
            'depth_anything_v2_large': 14
        }

        model_type = options["model_type"]
        
        model_id = None
        if isinstance(model_type, str):
            # Check if the string is in the available_models dictionary
            if model_type in available_models:
                model_id = available_models[model_type]
            else:
                available_strings = list(available_models.keys())
                raise HTTPException(status_code=400, detail={'error': 'Invalid model string', 'available_models': available_strings})
        elif isinstance(model_type, int):
            model_id = model_type
        else:
            raise HTTPException(status_code=400, detail={'error': 'Invalid model parameter type'})
        
        options["model_type"] = model_id

        video_parameters = options["video_parameters"]

        required_params = ["vid_numframes", "vid_fps", "vid_traj", "vid_shift", "vid_border", "dolly", "vid_format", "vid_ssaa", "output_filename"]
        
        missing_params = [param for param in required_params if param not in video_parameters]
        
        if missing_params:
            raise HTTPException(status_code=400, detail={'error': f"Missing required parameter(s): {', '.join(missing_params)}"})

        vid_numframes = video_parameters["vid_numframes"]
        vid_fps = video_parameters["vid_fps"]
        vid_traj = video_parameters["vid_traj"]
        vid_shift = video_parameters["vid_shift"]
        vid_border = video_parameters["vid_border"]
        dolly = video_parameters["dolly"]
        vid_format = video_parameters["vid_format"]
        vid_ssaa = int(video_parameters["vid_ssaa"])
        
        output_filename = video_parameters["output_filename"]
        output_path = os.path.dirname(output_filename)
        basename, extension = os.path.splitext(os.path.basename(output_filename))

        # Comparing video_format with the extension
        if vid_format != extension[1:]:
            raise HTTPException(status_code=400, detail={'error': f"Video format '{vid_format}' does not match with the extension '{extension}'."})

        pil_images = []
        for input_image in depth_input_images:
            pil_images.append(to_base64_PIL(input_image))
        outpath = backbone.get_outpath()

        mesh_fi_filename = video_parameters.get('mesh_fi_filename', None)

        if mesh_fi_filename and os.path.exists(mesh_fi_filename):
            mesh_fi = mesh_fi_filename
            print("Loaded existing mesh from: ", mesh_fi)
        else:
            # If there is no mesh file generate it.
            options["GEN_INPAINTED_MESH"] = True

            gen_obj = core_generation_funnel(outpath, pil_images, None, None, options)

            mesh_fi = None
            for count, type, result in gen_obj:
                if type == 'inpainted_mesh':
                    mesh_fi = result
                    break
                
            if mesh_fi:
                print("Created mesh in: ", mesh_fi)
            else:
                raise HTTPException(status_code=400, detail={'error': "The mesh has not been created"})

        run_makevideo(mesh_fi, vid_numframes, vid_fps, vid_traj, vid_shift, vid_border, dolly, vid_format, vid_ssaa, output_path, basename)

        return {"info": "Success"}


try:
    import modules.script_callbacks as script_callbacks
    if backbone.get_cmd_opt('api', False):
        script_callbacks.on_app_started(depth_api)
        print("Started the depthmap API. DO NOT HOST PUBLICLY - SECURITY RISKS!")
except:
    print('DepthMap API could not start')
