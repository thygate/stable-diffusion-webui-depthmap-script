import launch
import platform
import sys

# TODO: some dependencies apparently being reinstalled on every run. Investigate and fix.

if sys.version_info < (3, 8):
    launch.run_pip("install importlib-metadata", "importlib-metadata for depthmap script")
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
if not launch.is_installed('packaging'):
    launch.run_pip("install packaging", "packaging requirement for depthmap script")
from packaging.version import Version

def ensure(module_name, min_version=None):
    if launch.is_installed(module_name):
        if min_version is None or Version(importlib_metadata.version(module_name)) >= Version(min_version):
            return
    requirement = f'{module_name}>={min_version}' if min_version is not None else module_name
    cmd = f'install "{requirement}"'
    msg = f'{requirement} requirement for depthmap script'
    launch.run_pip(cmd, msg)

if not launch.is_installed("timm"): #0.6.7  #  For midas
    launch.run_pip('install --force-reinstall "timm==0.6.12"', "timm requirement for depthmap script")

ensure('matplotlib')

ensure('trimesh')

ensure('numba', '0.57.0')
ensure('vispy', '0.13.0')

ensure('rembg', '2.0.50')

if not launch.is_installed("moviepy"):
    launch.run_pip('install "moviepy==1.0.2"', "moviepy requirement for depthmap script")
ensure('transforms3d', '0.4.1')

ensure('imageio')  # 2.4.1
try:  # Dirty hack to not reinstall every time
    importlib_metadata.version('imageio-ffmpeg')
except:
    ensure('imageio-ffmpeg')


if not launch.is_installed("networkx"):
    launch.run_pip('install install "networkx==2.5"', "networkx requirement for depthmap script")
if platform.system() == 'Windows':
    ensure('pyqt5')

if platform.system() == 'Darwin':
    ensure('pyqt6')
