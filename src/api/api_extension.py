# Non-public API. Don't host publicly - SECURITY RISKS!
# (will only be on with --api starting option)
# Currently no API stability guarantees are provided - API may break on any new commit.

import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from PIL import Image
from itertools import tee
import json

import gradio as gr

from modules.api.models import List, Dict
from modules.api import api

from src.common_constants import GenerationOptions as go
from src.core import core_generation_funnel, CoreGenerationFunnelInp
from src import backbone
from src.misc import SCRIPT_VERSION
from src.api.api_constants import api_defaults, api_forced, api_options, models_to_index

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


def api_gen(input_images, client_options):

    default_options = CoreGenerationFunnelInp(api_defaults).values

    #TODO try-catch type errors here
    for key, value in client_options.items():
        if key == "model_type":
            default_options[key] = models_to_index(value)
            continue
        default_options[key] = value

    for key, value in api_forced.items():
        default_options[key.lower()] = value
        
    print(f"Processing {str(len(input_images))} images through the API")

    print(default_options)

    pil_images = []
    for input_image in input_images:
        pil_images.append(to_base64_PIL(input_image))
    outpath = backbone.get_outpath()
    gen_obj = core_generation_funnel(outpath, pil_images, None, None, default_options)
    return gen_obj

def depth_api(_: gr.Blocks, app: FastAPI):
    @app.get("/depth/version")
    async def version():
        return {"version": SCRIPT_VERSION}

    @app.get("/depth/get_options")
    async def get_options():
        return {
            "gen_options": [x.name.lower() for x in go],
            "api_options": api_options
        }

    @app.post("/depth/generate")
    async def process(
        input_images: List[str] = Body([], title='Input Images'),
        generate_options: Dict[str, object] = Body({}, title='Generation options', options= [x.name.lower() for x in go]),
        api_options: Dict[str, object] = Body({'outputs': ["depth"]}, title='Api options', options= api_options)
    ):
        
        if len(input_images)==0:
            raise HTTPException(status_code=422, detail="No images supplied")

        gen_obj = api_gen(input_images, generate_options)

        #NOTE Work around yield. (Might not be necessary, not sure if yield caches)
        _, gen_obj = tee (gen_obj)

        # If no outputs are specified assume depthmap is expected
        if len(api_options["outputs"])==0:
            api_options["outputs"] = ["depth"]

        results_based = {}
        for output_type in api_options["outputs"]:
            results_per_type = []

            for count, img_type, result in gen_obj:
                if img_type == output_type:
                    results_per_type.append( encode_to_base64(result) )
            
            # simpler output for simpler request.
            if api_options["outputs"] == ["depth"]:
                return {"images": results_per_type, "info": "Success"}

            if len(results_per_type)==0:
                results_based[output_type] = "Check options. no img-type of " + str(type) + " where generated"
            else:
                results_based[output_type] = results_per_type
        return {"images": results_based, "info": "Success"}
        