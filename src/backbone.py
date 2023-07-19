# DepthMap can be run inside stable-duiffusion-webui, but also separately.
# All the stable-duiffusion-webui stuff that the DepthMap relies on
# must be resided in this file (or in the scripts folder).

try:
    # stable-duiffusion-webui backbone
    from modules.images import save_image  # Should fail if not on stable-duiffusion-webui
    from modules.devices import torch_gc  # TODO: is this really sufficient?
    from modules.images import get_next_sequence_number
    from modules.call_queue import wrap_gradio_gpu_call
    from modules.shared import listfiles

    def get_opt(name, default):
        from modules.shared import opts
        if hasattr(opts, name):
            return opts.__getattr__(name)
        return default

    def get_cmd_opt(name, default):
        """Get command line argument"""
        from modules.shared import cmd_opts
        if hasattr(cmd_opts, name):
            return cmd_opts.__getattribute__(name)
        return default

    def gather_ops():
        """Parameters for depthmap generation"""
        from modules.shared import cmd_opts
        ops = {}
        if get_opt('depthmap_script_boost_rmax', None) is not None:
            ops['boost_whole_size_threshold'] = get_opt('depthmap_script_boost_rmax', None)
        ops['precision'] = cmd_opts.precision
        ops['no_half'] = cmd_opts.no_half
        return ops


    def get_outpath():
        """Get path where results are saved by default"""
        path = get_opt('outdir_samples', None)
        if path is None or len(path) == 0:
            path = get_opt('outdir_extras_samples', None)
        assert path is not None and len(path) > 0
        return path


    def unload_sd_model():
        from modules import shared, devices
        if shared.sd_model is not None:
            shared.sd_model.cond_stage_model.to(devices.cpu)
            shared.sd_model.first_stage_model.to(devices.cpu)


    def reload_sd_model():
        from modules import shared, devices
        if shared.sd_model is not None:
            shared.sd_model.cond_stage_model.to(devices.device)
            shared.sd_model.first_stage_model.to(devices.device)

    def get_hide_dirs():
        import modules.shared
        return modules.shared.hide_dirs
except:
    # Standalone backbone
    print("DepthMap did not detect stable-duiffusion-webui; launching with the standalone backbone.\n"
          "The standalone backbone is not on par with the stable-duiffusion-webui backbone.\n"
          "Some features may be missing or work differently. Please report bugs.\n")

    def save_image(image, path, basename, **kwargs):
        import os
        os.makedirs(path, exist_ok=True)
        fullfn = os.path.join(path, f"{get_next_sequence_number()}-{basename}.{kwargs['extension']}")
        image.save(fullfn, format=get_opt('samples_format', 'png'))

    def torch_gc():
        # TODO: is this really sufficient?
        import torch
        if torch.cuda.is_available():
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def get_next_sequence_number(outpath=None, basename=None):
        # Don't really care what the number will be... As long as it is unique.
        from datetime import datetime, timezone
        import random
        return int(f"{int(datetime.now(timezone.utc).timestamp())}{random.randint(1000,9999)}")

    def wrap_gradio_gpu_call(f): return f  # Displaying various stats is not supported

    def listfiles(dirname):
        import os
        filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname)) if not x.startswith(".")]
        return [file for file in filenames if os.path.isfile(file)]

    def get_opt(name, default): return default  # Configuring is not supported


    def get_cmd_opt(name, default): return default  # Configuring is not supported

    def gather_ops(): return {}  # Configuring is not supported

    def get_outpath(): return '.'

    def unload_sd_model(): pass  # Not needed

    def reload_sd_model(): pass  # Not needed

    def get_hide_dirs(): return {}  # Directories will not be hidden from traversal (except when starts with the dot)
