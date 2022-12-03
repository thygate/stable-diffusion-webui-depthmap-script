import launch
launch.git_clone("https://github.com/isl-org/MiDaS.git", "repositories/midas", "midas")
launch.git_clone("https://github.com/compphoto/BoostingMonocularDepth.git", "repositories/BoostingMonocularDepth", "BoostingMonocularDepth")
if not launch.is_installed("matplotlib"):
    launch.run_pip("install matplotlib", "requirements for depthmap script")