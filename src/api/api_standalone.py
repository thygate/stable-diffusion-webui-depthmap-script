from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI
from src.api.api_routes import depth_api

import gradio as gr

#TODO very primitive

#TODO add CORS

#TODO enable easy SSL. right now completely unsecured.

def init_api_no_webui():
    app = FastAPI()
    print("setting up endpoints")
    depth_api( gr.Blocks(), app)
    uvicorn.run('src.api.api_standalone:depth_api', port=7860, host="127.0.0.1")

def init_api(block, app):
    print("setting up endpoints")
    depth_api( block, app)