# This launches DepthMap without the AUTOMATIC1111/stable-diffusion-webui
# If DepthMap is installed as an extension,
# you may want to change the working directory to the stable-diffusion-webui root.

import argparse
import os
import pathlib
import builtins

import src.misc

def maybe_chdir():
    """Detects if DepthMap was installed as a stable-diffusion-webui script, but run without current directory set to
    the stable-diffusion-webui root. Changes current directory if needed, to aviod clutter."""
    try:
        file_path = pathlib.Path(__file__)
        path = file_path.parts
        while len(path) > 0 and path[-1] != src.misc.REPOSITORY_NAME:
            path = path[:-1]
        if len(path) >= 2 and path[-1] == src.misc.REPOSITORY_NAME and path[-2] == "extensions":
            path = path[:-2]
        listdir = os.listdir(str(pathlib.Path(*path)))
        if 'launch.py' in listdir and 'webui.py':
            os.chdir(str(pathlib.Path(**path)))
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", help="Create public link")
    parser.add_argument("--no_chdir", help="Do not try to use the root of stable-diffusion-webui")
    args = parser.parse_args()

    print(f"{src.misc.SCRIPT_FULL_NAME} running in standalone mode!")
    import src.common_ui
    if not args.no_chdir:
        maybe_chdir()
    src.common_ui.on_ui_tabs().launch(share=args.listen)
