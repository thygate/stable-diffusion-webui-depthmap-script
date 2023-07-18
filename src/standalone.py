# This launches Depth tab without the AUTOMATIC1111/stable-diffusion-webui
# Does not work yet.

import gradio as gr
import scripts.depthmap

demo = gr.Interface(fn=scripts.depthmap.on_ui_tabs, inputs="text", outputs="text")

demo.launch()
