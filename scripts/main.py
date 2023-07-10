import subprocess
import os
import pathlib
import torch

SCRIPT_NAME = "DepthMap"
SCRIPT_VERSION = "v0.3.13"

commit_hash = None  # TODO: understand why it would spam to stderr if changed to ... = get_commit_hash()
def get_commit_hash():
    global commit_hash
    if commit_hash is None:
        try:
            commit_hash = subprocess.check_output(
                [os.environ.get('GIT', "git"), "rev-parse", "HEAD"],
                cwd=pathlib.Path.cwd().joinpath('extensions/stable-diffusion-webui-depthmap-script/'),
                shell=False,
                stderr=subprocess.DEVNULL,
                encoding='utf8').strip()[0:8]
        except Exception:
            commit_hash = "<none>"
    return commit_hash


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
