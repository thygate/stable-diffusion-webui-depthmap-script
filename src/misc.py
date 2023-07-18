import subprocess
import os
import pathlib
import torch

def get_commit_hash():
    try:
        return subprocess.check_output(
            [os.environ.get("GIT", "git"), "rev-parse", "HEAD"],
            cwd=pathlib.Path.cwd().joinpath('extensions/stable-diffusion-webui-depthmap-script/'),
            shell=False,
            stderr=subprocess.DEVNULL,
            encoding='utf8').strip()[0:8]
    except Exception:
        return "<none>"


SCRIPT_NAME = "DepthMap"
SCRIPT_VERSION = "v0.4.1"
SCRIPT_FULL_NAME = f"{SCRIPT_NAME} {SCRIPT_VERSION} ({get_commit_hash()})"


def ensure_file_downloaded(filename, url, sha256_hash_prefix=None):
    # Do not check the hash every time - it is somewhat time-consuming
    if os.path.exists(filename):
        return

    if type(url) is not list:
        url = [url]
    for cur_url in url:
        try:
            print("Downloading", cur_url, "to", filename)
            torch.hub.download_url_to_file(cur_url, filename, sha256_hash_prefix)
            if os.path.exists(filename):
                return  # The correct model was downloaded, no need to try more
        except:
            pass
    raise RuntimeError('Download failed. Try again later or manually download the file to that location.')
