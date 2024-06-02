import subprocess
import os
import pathlib
import builtins

def get_commit_hash():
    try:
        file_path = pathlib.Path(__file__).parent
        return subprocess.check_output(
            [os.environ.get("GIT", "git"), "rev-parse", "HEAD"],
            cwd=file_path, shell=False, stderr=subprocess.DEVNULL, encoding='utf8').strip()[0:8]
    except Exception:
        return "<none>"


REPOSITORY_NAME = "stable-diffusion-webui-depthmap-script"
SCRIPT_NAME = "DepthMap"
SCRIPT_VERSION = "v0.4.6"
SCRIPT_FULL_NAME = f"{SCRIPT_NAME} {SCRIPT_VERSION} ({get_commit_hash()})"


# # Returns SHA256 hash of a file
# import hashlib
# def sha256sum(filename):
#     with open(filename, 'rb', buffering=0) as f:
#         return hashlib.file_digest(f, 'sha256').hexdigest()
def ensure_file_downloaded(filename, url, sha256_hash_prefix=None):
    import torch
    # Do not check the hash every time - it is somewhat time-consumin
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
    raise RuntimeError(f'Download failed. '
                       f'Try again later or manually download the file {filename} to location {url}.')
