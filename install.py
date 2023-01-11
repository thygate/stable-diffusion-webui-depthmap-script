import launch

if not launch.is_installed("timm"): #0.6.7
    launch.run_pip("install --force-reinstall timm==0.6.12", "timm requirement for depthmap script")

if not launch.is_installed("matplotlib"):
    launch.run_pip("install matplotlib", "matplotlib requirement for depthmap script")
    
if not launch.is_installed("numba"):
    launch.run_pip("install numba", "numba requirement for depthmap script")
if not launch.is_installed("vispy"):
    launch.run_pip("install vispy", "vispy requirement for depthmap script")

if not launch.is_installed("rembg"):
    launch.run_pip("install rembg", "rembg requirement for depthmap script")

if not launch.is_installed("moviepy"):
    launch.run_pip("install moviepy==1.0.2", "moviepy requirement for depthmap script")
if not launch.is_installed("transforms3d"):
    launch.run_pip("install transforms3d", "transforms3d requirement for depthmap script")
if not launch.is_installed("imageio"): #2.4.1
    launch.run_pip("install imageio", "imageio-ffmpeg requirement for depthmap script")
if not launch.is_installed("imageio-ffmpeg"):
    launch.run_pip("install imageio-ffmpeg", "imageio-ffmpeg requirement for depthmap script")
if not launch.is_installed("networkx"):
    launch.run_pip("install install networkx==2.5", "networkx requirement for depthmap script")
if not launch.is_installed("pyqt5"):
    launch.run_pip("install pyqt5", "pyqt5 requirement for depthmap script")

