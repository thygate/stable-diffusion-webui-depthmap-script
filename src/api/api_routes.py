# Non-public API. Don't host publicly - SECURITY RISKS!
# (will only be on with --api starting option)
# Currently no API stability guarantees are provided - API may break on any new commit.

from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from itertools import tee

import gradio as gr

from typing import Dict, List

from src.common_constants import GenerationOptions as go
from src.misc import SCRIPT_VERSION
from src.api.api_constants import api_options, models_to_index
from api.api_core import api_gen, encode_to_base64

def depth_api(_: gr.Blocks, app: FastAPI):
    @app.get("/depth/version")
    async def version():
        return {"version": SCRIPT_VERSION}

    @app.get("/depth/get_options")
    async def get_options():
        return {
            "gen_options": [x.name.lower() for x in go],
            "api_options": api_options,
            "model_names": models_to_index.keys()
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
        