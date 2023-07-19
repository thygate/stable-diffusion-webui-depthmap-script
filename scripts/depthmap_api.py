# Non-public API. Don't host publicly - SECURITY RISKS!
# (will only be on with --api starting option)
# Currently no API stability guarantees are provided - API may break on any new commit.

import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from PIL import Image

import gradio as gr

from modules.api.models import List, Dict
from modules.api import api

from src.core import core_generation_funnel
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
        print(f"Processing {str(len(depth_input_images))} images trough the API")

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


try:
    import modules.script_callbacks as script_callbacks
    if backbone.get_cmd_opt('api', False):
        script_callbacks.on_app_started(depth_api)
except:
    print('DepthMap API could not start')
