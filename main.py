# This launches DepthMap without the AUTOMATIC1111/stable-diffusion-webui

import argparse
import os
import pathlib

import src.misc


def maybe_chdir():
    """Detects if DepthMap was installed as a stable-diffusion-webui script, but run without current directory set to
    the stable-diffusion-webui root. Changes current directory if needed.
    This is to avoid re-downloading models and putting results into a wrong folder."""
    try:
        file_path = pathlib.Path(__file__)
        path = file_path.parts
        while len(path) > 0 and path[-1] != src.misc.REPOSITORY_NAME:
            path = path[:-1]
        if len(path) >= 2 and path[-1] == src.misc.REPOSITORY_NAME and path[-2] == "extensions":
            path = path[:-2]
        listdir = os.listdir(str(pathlib.Path(*path)))
        if 'launch.py' in listdir and 'webui.py':
            os.chdir(str(pathlib.Path(*path)))
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", help="Create public link", action='store_true')
    parser.add_argument("--listen", help="Create public link", action='store_true')
    parser.add_argument("--no_chdir", help="Do not try to use the root of stable-diffusion-webui", action='store_true')
    args = parser.parse_args()

    print(f"{src.misc.SCRIPT_FULL_NAME} running in standalone mode!")
    if not args.no_chdir:
        maybe_chdir()
    server_name = "0.0.0.0" if args.listen else None
    import src.common_ui
    src.common_ui.on_ui_tabs().launch(share=args.share, server_name=server_name)
