import launch
import platform
import sys

if sys.version_info < (3, 8):
    launch.run_pip("install importlib-metadata", "importlib-metadata for depthmap script")
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
if not launch.is_installed('packaging'):
    launch.run_pip("install packaging", "packaging requirement for depthmap script")
from packaging.version import Version

if not launch.is_installed("timm"): #0.6.7
    launch.run_pip('install --force-reinstall "timm==0.6.12"', "timm requirement for depthmap script")

if not launch.is_installed("matplotlib"):
    launch.run_pip("install matplotlib", "matplotlib requirement for depthmap script")

if not launch.is_installed("trimesh"):
    launch.run_pip("install trimesh", "requirements for depthmap script")
    
if not launch.is_installed("numba") or Version(importlib_metadata.version("numba")) < Version("0.57.0"):
    launch.run_pip('install "numba>=0.57.0"', "numba requirement for depthmap script")
if not launch.is_installed("vispy"):
    launch.run_pip("install vispy", "vispy requirement for depthmap script")

if not launch.is_installed("rembg"):
    launch.run_pip("install rembg", "rembg requirement for depthmap script")

if not launch.is_installed("moviepy"):
    launch.run_pip('install "moviepy==1.0.2"', "moviepy requirement for depthmap script")
if not launch.is_installed("transforms3d"):
    launch.run_pip("install transforms3d", "transforms3d requirement for depthmap script")
if not launch.is_installed("imageio"): #2.4.1
    launch.run_pip("install imageio", "imageio requirement for depthmap script")
if not launch.is_installed("imageio-ffmpeg"):
    launch.run_pip("install imageio-ffmpeg", "imageio-ffmpeg requirement for depthmap script")
if not launch.is_installed("networkx"):
    launch.run_pip('install install "networkx==2.5"', "networkx requirement for depthmap script")
if platform.system() == 'Windows':
    if not launch.is_installed("pyqt5"):
        launch.run_pip("install pyqt5", "pyqt5 requirement for depthmap script")

if platform.system() == 'Darwin':
    if not launch.is_installed("pyqt6"):
        launch.run_pip("install pyqt6", "pyqt6 requirement for depthmap script")
