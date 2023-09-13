# Non-public API. Don't host publicly - SECURITY RISKS!
# (will only be on with --api starting option)
# Currently no API stability guarantees are provided - API may break on any new commit.

from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException

from typing import Dict, List
from PIL import Image

from src.common_constants import GenerationOptions as go
from src.misc import SCRIPT_VERSION
from src.api.api_constants import api_options, models_to_index
from src.api.api_core import api_gen, encode_to_base64

# _ parameter is needed for auto1111 extensions (_ is type gr.Blocks)
def depth_api(_, app: FastAPI):
    @app.get("/depth/version")
    async def version():
        return {"version": SCRIPT_VERSION}

    @app.get("/depth/get_options")
    async def get_options():
        return {
            "gen_options": [x.name.lower() for x in go],
            "api_options": list(api_options.keys()),
            "model_names": list(models_to_index.keys())
        }

    @app.post("/depth/generate")
    async def process(
        input_images: List[str] = Body([], title='Input Images'),
        generate_options: Dict[str, object] = Body({}, title='Generation options', options= [x.name.lower() for x in go]),
        api_options: Dict[str, object] = Body({}, title='Api options', options= api_options)
    ):
        
        if len(input_images)==0:
            raise HTTPException(status_code=422, detail="No images supplied")

        gen_obj = api_gen(input_images, generate_options)

        results_based = []
        for count, type, result in gen_obj:
            if not isinstance(result, Image.Image):
                continue
            results_based += [encode_to_base64(result)]
        return {"images": results_based, "info": "Success"}
        
        