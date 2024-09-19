# DepthMap can be run inside stable-diffusion-webui, but also separately.
# All the stable-diffusion-webui stuff that the DepthMap relies on
# must be resided in this file (or in the scripts folder).
import pathlib
from datetime import datetime
import enum
import sys


class BackboneType(enum.Enum):
    WEBUI = 1
    STANDALONE = 2


try:
    # stable-diffusion-webui backbone
    from modules.images import save_image  # Should fail if not on stable-diffusion-webui
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
        ops = {}
        for s in ['boost_rmax', 'precision', 'no_half', 'marigold_ensembles', 'marigold_steps']:
            c = get_opt('depthmap_script_' + s, None)
            if c is None:
                c = get_cmd_opt(s, None)
            if c is not None:
                ops[s] = c
        # sanitize for integers.
        for s in ['marigold_ensembles', 'marigold_steps']:
            if s in ops:
                ops[s] = int(ops[s])
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
        try:
            if shared.sd_model is not None:
                if shared.sd_model.cond_stage_model is not None:
                    shared.sd_model.cond_stage_model.to(devices.cpu)
                if shared.sd_model.first_stage_model is not None:
                    shared.sd_model.first_stage_model.to(devices.cpu)
        except Exception as e:
            from backend import memory_management
            print('trying to catch forge (might be a attribute error)')
            if type(e)== AttributeError:
                memory_management.unload_all_models()
                memory_management.soft_empty_cache()
            else:
                raise
        # Maybe something else???


    def reload_sd_model():
        from modules import shared, devices
        try:
            if shared.sd_model is not None:
                if shared.sd_model.cond_stage_model is not None:
                    shared.sd_model.cond_stage_model.to(devices.device)
                if shared.sd_model.first_stage_model:
                    shared.sd_model.first_stage_model.to(devices.device)
        except Exception as e:
            from backend import memory_management
            print('trying to catch forge (might be a attribute error)')
            if type(e)== AttributeError:
                memory_management.unload_all_models()
                memory_management.soft_empty_cache()
            else:
                raise
        # Maybe something else???

    def get_hide_dirs():
        import modules.shared
        return modules.shared.hide_dirs

    USED_BACKBONE = BackboneType.WEBUI
except:
    # Standalone backbone
    print(  # "  DepthMap did not detect stable-diffusion-webui; launching with the standalone backbone.\n"
          "  The standalone mode is not on par with the stable-diffusion-webui mode.\n"
          "  Some features may be missing or work differently. Please report bugs.\n")

    def save_image(image, path, basename, **kwargs):
        import os
        os.makedirs(path, exist_ok=True)
        if 'suffix' not in kwargs or len(kwargs['suffix']) == 0:
            kwargs['suffix'] = ''
        else:
            kwargs['suffix'] = f"-{kwargs['suffix']}"
        format = get_opt('samples_format', kwargs['extension'])
        fullfn = os.path.join(
            path, f"{basename}-{get_next_sequence_number(path, basename)}{kwargs['suffix']}.{format}")
        image.save(fullfn, format=format)

    def torch_gc():
        # TODO: is this really sufficient?
        import torch
        if torch.cuda.is_available():
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    launched_at = int(datetime.now().timestamp())
    backbone_current_seq_number = 0

    # Make sure to preserve the function signature when calling!
    def get_next_sequence_number(outpath, basename):
        global backbone_current_seq_number
        backbone_current_seq_number += 1
        return int(f"{launched_at}{backbone_current_seq_number:04}")

    def wrap_gradio_gpu_call(f): return f  # Displaying various stats is not supported

    def listfiles(dirname):
        import os
        filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname)) if not x.startswith(".")]
        return [file for file in filenames if os.path.isfile(file)]

    def get_opt(name, default): return default  # Configuring is not supported


    def get_cmd_opt(name, default): return default  # Configuring is not supported

    def gather_ops():  # Configuring is not supported
        return {'boost_rmax': 1600,
                'precision': 'autocast',
                'no_half': False,
                'marigold_ensembles': 5,
                'marigold_steps': 12}

    def get_outpath(): return str(pathlib.Path('.', 'outputs'))

    def unload_sd_model(): pass  # Not needed

    def reload_sd_model(): pass  # Not needed

    def get_hide_dirs(): return {}  # Directories will not be hidden from traversal (except when starts with the dot)


    USED_BACKBONE = BackboneType.STANDALONE
