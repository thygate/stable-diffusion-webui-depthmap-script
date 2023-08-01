from fastapi import FastAPI
import uvicorn
from src.api.api_routes import depth_api

# without gradio
def init_api_no_webui():
    app = FastAPI()
    print("setting up api endpoints")
    depth_api( '', app)
    print("api running")
    uvicorn.run('src.api.api_standalone:depth_api', port=7860, host="127.0.0.1")

def init_api(block, app):
    print("setting up api endpoints")
    depth_api( block, app)
    print("api running")