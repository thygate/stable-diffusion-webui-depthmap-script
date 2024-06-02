# Installs dependencies
# Make sure to add to requirements.txt - it can be used for the standalone mode

import launch
import platform
import sys
import importlib.metadata

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


ensure('timm', '0.9.2')  # For midas, specified just in case

ensure('matplotlib')

ensure('trimesh')

ensure('numba', '0.57.0')
ensure('vispy', '0.13.0')

ensure('rembg', '2.0.50')

if not launch.is_installed("moviepy"):
    launch.run_pip('install "moviepy==1.0.2"', "moviepy requirement for depthmap script")
ensure('transforms3d', '0.4.1')

ensure('diffusers', '0.20.1')  # For Merigold

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
    ensure('PyOpenGL', '3.1.7')

# Depth Anything
def get_installed_version(package: str):
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None
def try_install_from_wheel(pkg_name: str, wheel_url: str):
    if get_installed_version(pkg_name) is not None:
        return
    try:
        launch.run_pip(f"install {wheel_url}", f" {pkg_name} requirement for depthmap script")
    except Exception as e:
        print('Failed to install wheel for Depth Anything support. It won\'t work.')
try_install_from_wheel(
    "depth_anything",
    "https://github.com/huchenlei/Depth-Anything/releases/download/v1.0.0/depth_anything-2024.1.22.0-py2.py3-none-any.whl")
