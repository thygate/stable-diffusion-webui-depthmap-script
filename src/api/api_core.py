import numpy as np
from PIL import PngImagePlugin, Image
import base64
from io import BytesIO
from fastapi.exceptions import HTTPException

import gradio as gr


from src.core import core_generation_funnel, CoreGenerationFunnelInp
from src import backbone
from src.api.api_constants import api_defaults, api_forced, models_to_index

# moedified from modules/api/api.py auto1111
def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e

# modified from modules/api/api.py auto1111. TODO check that internally we always use png. Removed webp and jpeg
def encode_pil_to_base64(image, image_type='png'):
    with BytesIO() as output_bytes:

        if image_type == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None))

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)

def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return encode_pil_to_base64(image)
    elif type(image) is np.ndarray:
        return encode_np_to_base64(image)
    else:
        return ""

def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return encode_pil_to_base64(pil)

def to_base64_PIL(encoding: str):
    return Image.fromarray(np.array(decode_base64_to_image(encoding)).astype('uint8'))


def api_gen(input_images, client_options):

    default_options = CoreGenerationFunnelInp(api_defaults).values

    #TODO try-catch type errors here
    for key, value in client_options.items():
        if key == "model_type":
            default_options[key] = models_to_index[value]
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