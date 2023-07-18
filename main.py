# This launches DepthMap without the AUTOMATIC1111/stable-diffusion-webui
# If DepthMap is installed as an extension,
# you may want to change the working directory to the stable-diffusion-webui root.

import argparse
import src.common_ui

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", help="Create public link")
    args = parser.parse_args()

    src.common_ui.on_ui_tabs().launch(share=args.listen)
