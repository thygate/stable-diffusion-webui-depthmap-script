from pathlib import Path
from PIL import Image

try:
    from tqdm import trange
except:
    from builtins import range as trange

import torch, gc
import cv2
import os.path
import numpy as np
import copy
import platform
import math
import traceback

# Our code
from src.misc import *
from src.stereoimage_generation import create_stereoimages
from src.normalmap_generation import create_normalmap
from src.depthmap_generation import ModelHolder
from src import backbone

# 3d-photo-inpainting imports
from inpaint.mesh import write_mesh, read_mesh, output_3d_photo
from inpaint.networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from inpaint.utils import path_planning
from inpaint.bilateral_filtering import sparse_bilateral_filtering

global video_mesh_data, video_mesh_fn
video_mesh_data = None
video_mesh_fn = None

model_holder = ModelHolder()


def convert_i16_to_rgb(image, like):
    # three channel, 8 bits per channel image
    output = np.zeros_like(like)
    output[:, :, 0] = image / 256.0
    output[:, :, 1] = image / 256.0
    output[:, :, 2] = image / 256.0
    return output


def core_generation_funnel(outpath, inputimages, inputdepthmaps, inputnames, inp, ops=None, **kwargs):
    if len(inputimages) == 0 or inputimages[0] is None:
        return [], '', ''
    if inputdepthmaps is None or len(inputdepthmaps) == 0:
        inputdepthmaps: list[Image] = [None for _ in range(len(inputimages))]
    inputdepthmaps_complete = all([x is not None for x in inputdepthmaps])

    background_removal = inp["background_removal"]
    background_removal_model = inp["background_removal_model"]
    boost = inp["boost"]
    clipdepth = inp["clipdepth"]
    clipthreshold_far = inp["clipthreshold_far"]
    clipthreshold_near = inp["clipthreshold_near"]
    combine_output = inp["combine_output"]
    combine_output_axis = inp["combine_output_axis"]
    depthmap_compute_device = inp["compute_device"]
    gen_mesh = inp["gen_mesh"]
    gen_normalmap = inp["gen_normalmap"] if "gen_normalmap" in inp else False
    gen_stereo = inp["gen_stereo"]
    inpaint = inp["inpaint"] if "inpaint" in inp else False
    inpaint_vids = inp["inpaint_vids"] if "inpaint_vids" in inp else False
    invert_depth = inp["invert_depth"]
    match_size = inp["match_size"]
    mesh_occlude = inp["mesh_occlude"]
    mesh_spherical = inp["mesh_spherical"]
    model_type = inp["model_type"]
    net_height = inp["net_height"]
    net_width = inp["net_width"]
    normalmap_pre_blur = inp["normalmap_pre_blur"]
    normalmap_pre_blur_kernel = inp["normalmap_pre_blur_kernel"]
    normalmap_sobel = inp["normalmap_sobel"]
    normalmap_sobel_kernel = inp["normalmap_sobel_kernel"]
    normalmap_post_blur = inp["normalmap_post_blur"]
    normalmap_post_blur_kernel = inp["normalmap_post_blur_kernel"]
    normalmap_invert = inp['normalmap_invert']
    pre_depth_background_removal = inp["pre_depth_background_removal"]
    save_background_removal_masks = inp["save_background_removal_masks"]
    output_depth = inp["output_depth"]
    show_heat = inp["show_heat"]
    stereo_balance = inp["stereo_balance"]
    stereo_divergence = inp["stereo_divergence"]
    stereo_fill = inp["stereo_fill"]
    stereo_modes = inp["stereo_modes"]
    stereo_separation = inp["stereo_separation"]

    if ops is None:
        ops = {}
    model_holder.update_settings(**ops)

    # TODO: ideally, run_depthmap should not save meshes - that makes the function not pure
    print(SCRIPT_FULL_NAME)

    backbone.unload_sd_model()

    # TODO: this still should not be here
    background_removed_images = []
    # remove on base image before depth calculation
    if background_removal:
        if pre_depth_background_removal:
            inputimages = batched_background_removal(inputimages, background_removal_model)
            background_removed_images = inputimages
        else:
            background_removed_images = batched_background_removal(inputimages, background_removal_model)

    # init torch device
    if depthmap_compute_device == 'GPU' and not torch.cuda.is_available():
        print('WARNING: Cuda device was not found, cpu will be used')
        depthmap_compute_device = 'CPU'
    if depthmap_compute_device == 'GPU':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device: %s" % device)

    generated_images = [{} for _ in range(len(inputimages))]
    """Images that will be returned.
    Every array element corresponds to particular input image.
    Dictionary keys are types of images that were derived from the input image."""
    # TODO: ???
    meshsimple_fi = None
    inpaint_imgs = []
    inpaint_depths = []

    try:
        if not inputdepthmaps_complete:
            print("Loading model(s) ..")
            model_holder.ensure_models(model_type, device, boost)
        model = model_holder.depth_model
        pix2pix_model = model_holder.pix2pix_model

        print("Computing output(s) ..")
        # iterate over input images
        for count in trange(0, len(inputimages)):
            # Convert single channel input (PIL) images to rgb
            if inputimages[count].mode == 'I':
                inputimages[count].point(lambda p: p * 0.0039063096, mode='RGB')
                inputimages[count] = inputimages[count].convert('RGB')

            raw_prediction = None
            """Raw prediction, as returned by a model. None if input depthmap is used."""
            raw_prediction_invert = False
            """True if near=dark on raw_prediction"""
            out = None
            if inputdepthmaps is not None and inputdepthmaps[count] is not None:
                # use custom depthmap
                dimg = inputdepthmaps[count]
                # resize if not same size as input
                if dimg.width != inputimages[count].width or dimg.height != inputimages[count].height:
                    dimg = dimg.resize((inputimages[count].width, inputimages[count].height), Image.Resampling.LANCZOS)

                if dimg.mode == 'I' or dimg.mode == 'P' or dimg.mode == 'L':
                    out = np.asarray(dimg, dtype="float")
                else:
                    out = np.asarray(dimg, dtype="float")[:, :, 0]
            else:
                # override net size (size may be different for different images)
                if match_size:
                    # Round up to a multiple of 32 to avoid potential issues
                    net_width = (inputimages[count].width + 31) // 32 * 32
                    net_height = (inputimages[count].height + 31) // 32 * 32
                raw_prediction, raw_prediction_invert = \
                    model_holder.get_raw_prediction(inputimages[count], net_width, net_height)

                # output
                if abs(raw_prediction.max() - raw_prediction.min()) > np.finfo("float").eps:
                    out = np.copy(raw_prediction)
                    # TODO: some models may output negative values, maybe these should be clamped to zero.
                    if raw_prediction_invert:
                        out *= -1
                    if clipdepth:
                        out = (out - out.min()) / (out.max() - out.min())  # normalize to [0; 1]
                        out = np.clip(out, clipthreshold_far, clipthreshold_near)
                else:
                    # Regretfully, the depthmap is broken and will be replaced with a black image
                    out = np.zeros(raw_prediction.shape)
            out = (out - out.min()) / (out.max() - out.min())  # normalize to [0; 1]

            # Single channel, 16 bit image. This loses some precision!
            # uint16 conversion uses round-down, therefore values should be [0; 2**16)
            numbytes = 2
            max_val = (2 ** (8 * numbytes))
            out = np.clip(out * max_val, 0, max_val - 0.1)  # Clipping form above is needed to avoid overflowing
            img_output = out.astype("uint16")
            """Depthmap (near=bright), as uint16"""

            # if 3dinpainting, store maps for processing in second pass
            if inpaint:
                inpaint_imgs.append(inputimages[count])
                inpaint_depths.append(img_output)

            # applying background masks after depth
            if background_removal:
                print('applying background masks')
                background_removed_image = background_removed_images[count]
                # maybe a threshold cut would be better on the line below.
                background_removed_array = np.array(background_removed_image)
                bg_mask = (background_removed_array[:, :, 0] == 0) & (background_removed_array[:, :, 1] == 0) & (
                        background_removed_array[:, :, 2] == 0) & (background_removed_array[:, :, 3] <= 0.2)
                img_output[bg_mask] = 0  # far value

                generated_images[count]['background_removed'] = background_removed_image

                if save_background_removal_masks:
                    bg_array = (1 - bg_mask.astype('int8')) * 255
                    mask_array = np.stack((bg_array, bg_array, bg_array, bg_array), axis=2)
                    mask_image = Image.fromarray(mask_array.astype(np.uint8))

                    generated_images[count]['foreground_mask'] = mask_image

            # A weird quirk: if user tries to save depthmap, whereas input depthmap is used,
            # depthmap will be outputed, even if combine_output is used.
            if output_depth and inputdepthmaps[count] is None:
                if output_depth:
                    img_depth = cv2.bitwise_not(img_output) if invert_depth else img_output
                    if combine_output:
                        img_concat = Image.fromarray(np.concatenate(
                            (inputimages[count], convert_i16_to_rgb(img_depth, inputimages[count])),
                            axis=combine_output_axis))
                        generated_images[count]['concat_depth'] = img_concat
                    else:
                        generated_images[count]['depth'] = Image.fromarray(img_depth)

            if gen_stereo:
                print("Generating stereoscopic images..")
                stereoimages = create_stereoimages(inputimages[count], img_output, stereo_divergence, stereo_separation,
                                                   stereo_modes, stereo_balance, stereo_fill)
                for c in range(0, len(stereoimages)):
                    generated_images[count][stereo_modes[c]] = stereoimages[c]

            if gen_normalmap:
                generated_images[count]['normalmap'] = create_normalmap(
                    img_output,
                    normalmap_pre_blur_kernel if normalmap_pre_blur else None,
                    normalmap_sobel_kernel if normalmap_sobel else None,
                    normalmap_post_blur_kernel if normalmap_post_blur else None,
                    normalmap_invert
                )

            if show_heat:
                from dzoedepth.utils.misc import colorize
                heatmap = Image.fromarray(colorize(img_output, cmap='inferno'))
                generated_images[count]['heatmap'] = heatmap

            # gen mesh
            if gen_mesh:
                print(f"\nGenerating (occluded) mesh ..")
                basename = 'depthmap'
                meshsimple_fi = get_uniquefn(outpath, basename, 'obj')
                meshsimple_fi = os.path.join(outpath, meshsimple_fi + '_simple.obj')

                depthi = raw_prediction if raw_prediction is not None else out
                depthi_min, depthi_max = depthi.min(), depthi.max()
                # try to map output to sensible values for non zoedepth models, boost, or custom maps
                if model_type not in [7, 8, 9] or boost or inputdepthmaps[count] is not None:
                    # invert if midas
                    if model_type > 0 or inputdepthmaps[count] is not None:  # TODO: Weird
                        depthi = depthi_max - depthi + depthi_min
                        depth_max = depthi.max()
                        depth_min = depthi.min()
                    # make positive
                    if depthi_min < 0:
                        depthi = depthi - depthi_min
                        depth_max = depthi.max()
                        depth_min = depthi.min()
                    # scale down
                    if depthi.max() > 10.0:
                        depthi = 4.0 * (depthi - depthi_min) / (depthi_max - depthi_min)
                    # offset
                    depthi = depthi + 1.0

                mesh = create_mesh(inputimages[count], depthi, keep_edges=not mesh_occlude, spherical=mesh_spherical)
                mesh.export(meshsimple_fi)
            
            if "save_img" in kwargs:
                kwargs.get("save_img")(count, generated_images[count])

        print("Computing output(s) done.")
    except RuntimeError as e:
        # TODO: display in UI
        if 'out of memory' in str(e):
            suggestion = "ERROR: out of memory, could not generate depthmap!\nPlease try a different model"
            if device != torch.device("cpu"):
                suggestion += ", or try using the CPU"
            if boost:
                suggestion += ", or disable BOOST"
            print(f"{suggestion}.")
        else:
            raise e
    finally:
        if backbone.get_opt('depthmap_script_keepmodels', True):
            model_holder.offload()  # Swap to CPU memory
        else:
            if 'model' in locals():
                del model
            if 'pix2pixmodel' in locals():
                del pix2pix_model
            model_holder.unload_models()

        gc.collect()
        backbone.torch_gc()

    # TODO: This should not be here
    mesh_fi = None
    if inpaint:
        try:
            mesh_fi = run_3dphoto(device, inpaint_imgs, inpaint_depths, inputnames, outpath, inpaint_vids, 1, "mp4")
        except Exception as e:
            print(f'{str(e)}, some issue with generating inpainted mesh')

    backbone.reload_sd_model()
    print("All done.\n")
    return generated_images, mesh_fi, meshsimple_fi


def get_uniquefn(outpath, basename, ext):
    # Inefficient and may fail, maybe use unbounded binary search?
    basecount = backbone.get_next_sequence_number(outpath, basename)
    if basecount > 0: basecount = basecount - 1
    fullfn = None
    for i in range(500):
        fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
        fullfn = os.path.join(outpath, f"{fn}.{ext}")
        if not os.path.exists(fullfn):
            break
    basename = Path(fullfn).stem

    return basename


def run_3dphoto(device, img_rgb, img_depth, inputnames, outpath, inpaint_vids, vid_ssaa, vid_format):
    mesh_fi = ''
    try:
        print("Running 3D Photo Inpainting .. ")
        edgemodel_path = './models/3dphoto/edge_model.pth'
        depthmodel_path = './models/3dphoto/depth_model.pth'
        colormodel_path = './models/3dphoto/color_model.pth'
        # create paths to model if not present
        os.makedirs('./models/3dphoto/', exist_ok=True)

        ensure_file_downloaded(edgemodel_path,
                               "https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth")
        ensure_file_downloaded(depthmodel_path,
                               "https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth")
        ensure_file_downloaded(colormodel_path,
                               "https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth")

        print("Loading edge model ..")
        depth_edge_model = Inpaint_Edge_Net(init_weights=True)
        depth_edge_weight = torch.load(edgemodel_path, map_location=torch.device(device))
        depth_edge_model.load_state_dict(depth_edge_weight)
        depth_edge_model = depth_edge_model.to(device)
        depth_edge_model.eval()
        print("Loading depth model ..")
        depth_feat_model = Inpaint_Depth_Net()
        depth_feat_weight = torch.load(depthmodel_path, map_location=torch.device(device))
        depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
        depth_feat_model = depth_feat_model.to(device)
        depth_feat_model.eval()
        depth_feat_model = depth_feat_model.to(device)
        print("Loading rgb model ..")
        rgb_model = Inpaint_Color_Net()
        rgb_feat_weight = torch.load(colormodel_path, map_location=torch.device(device))
        rgb_model.load_state_dict(rgb_feat_weight)
        rgb_model.eval()
        rgb_model = rgb_model.to(device)

        config = {}
        config["gpu_ids"] = 0
        config['extrapolation_thickness'] = 60
        config['extrapolate_border'] = True
        config['depth_threshold'] = 0.04
        config['redundant_number'] = 12
        config['ext_edge_threshold'] = 0.002
        config['background_thickness'] = 70
        config['context_thickness'] = 140
        config['background_thickness_2'] = 70
        config['context_thickness_2'] = 70
        config['log_depth'] = True
        config['depth_edge_dilate'] = 10
        config['depth_edge_dilate_2'] = 5
        config['largest_size'] = 512
        config['repeat_inpaint_edge'] = True
        config['ply_fmt'] = "bin"

        config['save_ply'] = backbone.get_opt('depthmap_script_save_ply', False)
        config['save_obj'] = True

        if device == torch.device("cpu"):
            config["gpu_ids"] = -1

        for count in trange(0, len(img_rgb)):
            basename = 'depthmap'
            if inputnames is not None:
                if inputnames[count] is not None:
                    p = Path(inputnames[count])
                    basename = p.stem

            basename = get_uniquefn(outpath, basename, 'obj')
            mesh_fi = os.path.join(outpath, basename + '.obj')

            print(f"\nGenerating inpainted mesh .. (go make some coffee) ..")

            # from inpaint.utils.get_MiDaS_samples
            W = img_rgb[count].width
            H = img_rgb[count].height
            int_mtx = np.array([[max(H, W), 0, W // 2], [0, max(H, W), H // 2], [0, 0, 1]]).astype(np.float32)
            if int_mtx.max() > 1:
                int_mtx[0, :] = int_mtx[0, :] / float(W)
                int_mtx[1, :] = int_mtx[1, :] / float(H)

            # how inpaint.utils.read_MiDaS_depth() imports depthmap
            disp = img_depth[count].astype(np.float32)
            disp = disp - disp.min()
            disp = cv2.blur(disp / disp.max(), ksize=(3, 3)) * disp.max()
            disp = (disp / disp.max()) * 3.0
            depth = 1. / np.maximum(disp, 0.05)

            # rgb input
            img = np.asarray(img_rgb[count])

            # run sparse bilateral filter
            config['sparse_iter'] = 5
            config['filter_size'] = [7, 7, 5, 5, 5]
            config['sigma_s'] = 4.0
            config['sigma_r'] = 0.5
            vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), img.copy(), config,
                                                                num_iter=config['sparse_iter'], spdb=False)
            depth = vis_depths[-1]

            # bilat_fn = os.path.join(outpath, basename +'_bilatdepth.png')
            # cv2.imwrite(bilat_fn, depth)

            rt_info = write_mesh(img,
                                 depth,
                                 int_mtx,
                                 mesh_fi,
                                 config,
                                 rgb_model,
                                 depth_edge_model,
                                 depth_edge_model,
                                 depth_feat_model)

            if rt_info is not False and inpaint_vids:
                run_3dphoto_videos(mesh_fi, basename, outpath, 300, 40,
                                   [0.03, 0.03, 0.05, 0.03],
                                   ['double-straight-line', 'double-straight-line', 'circle', 'circle'],
                                   [0.00, 0.00, -0.015, -0.015],
                                   [0.00, 0.00, -0.015, -0.00],
                                   [-0.05, -0.05, -0.05, -0.05],
                                   ['dolly-zoom-in', 'zoom-in', 'circle', 'swing'], False, vid_format, vid_ssaa)

            backbone.torch_gc()

    finally:
        del rgb_model
        rgb_model = None
        del depth_edge_model
        depth_edge_model = None
        del depth_feat_model
        depth_feat_model = None
        backbone.torch_gc()

    return mesh_fi


def run_3dphoto_videos(mesh_fi, basename, outpath, num_frames, fps, crop_border, traj_types, x_shift_range,
                       y_shift_range, z_shift_range, video_postfix, vid_dolly, vid_format, vid_ssaa):
    import vispy
    if platform.system() == 'Windows':
        vispy.use(app='PyQt5')
    elif platform.system() == 'Darwin':
        vispy.use('PyQt6')
    else:
        vispy.use(app='egl')

    # read ply
    global video_mesh_data, video_mesh_fn
    if video_mesh_fn is None or video_mesh_fn != mesh_fi:
        del video_mesh_data
        video_mesh_fn = mesh_fi
        video_mesh_data = read_mesh(mesh_fi)

    verts, colors, faces, Height, Width, hFov, vFov, mean_loc_depth = video_mesh_data

    original_w = output_w = W = Width
    original_h = output_h = H = Height
    int_mtx = np.array([[max(H, W), 0, W // 2], [0, max(H, W), H // 2], [0, 0, 1]]).astype(np.float32)
    if int_mtx.max() > 1:
        int_mtx[0, :] = int_mtx[0, :] / float(W)
        int_mtx[1, :] = int_mtx[1, :] / float(H)

    config = {}
    config['video_folder'] = outpath
    config['num_frames'] = num_frames
    config['fps'] = fps
    config['crop_border'] = crop_border
    config['traj_types'] = traj_types
    config['x_shift_range'] = x_shift_range
    config['y_shift_range'] = y_shift_range
    config['z_shift_range'] = z_shift_range
    config['video_postfix'] = video_postfix
    config['ssaa'] = vid_ssaa

    # from inpaint.utils.get_MiDaS_samples
    generic_pose = np.eye(4)
    assert len(config['traj_types']) == len(config['x_shift_range']) == \
           len(config['y_shift_range']) == len(config['z_shift_range']) == len(config['video_postfix']), \
        "The number of elements in 'traj_types', 'x_shift_range', 'y_shift_range', 'z_shift_range' and \
            'video_postfix' should be equal."
    tgt_pose = [[generic_pose * 1]]
    tgts_poses = []
    for traj_idx in range(len(config['traj_types'])):
        tgt_poses = []
        sx, sy, sz = path_planning(config['num_frames'], config['x_shift_range'][traj_idx],
                                   config['y_shift_range'][traj_idx],
                                   config['z_shift_range'][traj_idx], path_type=config['traj_types'][traj_idx])
        for xx, yy, zz in zip(sx, sy, sz):
            tgt_poses.append(generic_pose * 1.)
            tgt_poses[-1][:3, -1] = np.array([xx, yy, zz])
        tgts_poses += [tgt_poses]
    tgt_pose = generic_pose * 1

    # seems we only need the depthmap to calc mean_loc_depth, which is only used when doing 'dolly'
    # width and height are already in the ply file in the comments ..
    # might try to add the mean_loc_depth to it too
    # did just that
    # mean_loc_depth = img_depth[img_depth.shape[0]//2, img_depth.shape[1]//2]

    print("Generating videos ..")

    normal_canvas, all_canvas = None, None
    videos_poses, video_basename = copy.deepcopy(tgts_poses), basename
    top = (original_h // 2 - int_mtx[1, 2] * output_h)
    left = (original_w // 2 - int_mtx[0, 2] * output_w)
    down, right = top + output_h, left + output_w
    border = [int(xx) for xx in [top, down, left, right]]
    normal_canvas, all_canvas, fn_saved = output_3d_photo(verts.copy(), colors.copy(), faces.copy(),
                                                          copy.deepcopy(Height), copy.deepcopy(Width),
                                                          copy.deepcopy(hFov), copy.deepcopy(vFov),
                                                          copy.deepcopy(tgt_pose), config['video_postfix'],
                                                          copy.deepcopy(generic_pose),
                                                          copy.deepcopy(config['video_folder']),
                                                          None, copy.deepcopy(int_mtx), config, None,
                                                          videos_poses, video_basename, original_h, original_w,
                                                          border=border, depth=None, normal_canvas=normal_canvas,
                                                          all_canvas=all_canvas,
                                                          mean_loc_depth=mean_loc_depth, dolly=vid_dolly,
                                                          fnExt=vid_format)
    return fn_saved


# called from gen vid tab button
def run_makevideo(fn_mesh, vid_numframes, vid_fps, vid_traj, vid_shift, vid_border, dolly, vid_format, vid_ssaa):
    if len(fn_mesh) == 0 or not os.path.exists(fn_mesh):
        raise Exception("Could not open mesh.")

    vid_ssaa = int(vid_ssaa)

    # traj type
    if vid_traj == 0:
        vid_traj = ['straight-line']
    elif vid_traj == 1:
        vid_traj = ['double-straight-line']
    elif vid_traj == 2:
        vid_traj = ['circle']

    num_fps = int(vid_fps)
    num_frames = int(vid_numframes)
    shifts = vid_shift.split(',')
    if len(shifts) != 3:
        raise Exception("Translate requires 3 elements.")
    x_shift_range = [float(shifts[0])]
    y_shift_range = [float(shifts[1])]
    z_shift_range = [float(shifts[2])]

    borders = vid_border.split(',')
    if len(borders) != 4:
        raise Exception("Crop Border requires 4 elements.")
    crop_border = [float(borders[0]), float(borders[1]), float(borders[2]), float(borders[3])]

    # output path and filename mess ..
    basename = Path(fn_mesh).stem
    outpath = backbone.get_outpath()
    # unique filename
    basecount = backbone.get_next_sequence_number(outpath, basename)
    if basecount > 0: basecount = basecount - 1
    fullfn = None
    for i in range(500):
        fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
        fullfn = os.path.join(outpath, f"{fn}_." + vid_format)
        if not os.path.exists(fullfn):
            break
    basename = Path(fullfn).stem
    basename = basename[:-1]

    print("Loading mesh ..")

    fn_saved = run_3dphoto_videos(fn_mesh, basename, outpath, num_frames, num_fps, crop_border, vid_traj, x_shift_range,
                                  y_shift_range, z_shift_range, [''], dolly, vid_format, vid_ssaa)

    return fn_saved[-1], fn_saved[-1], ''


def unload_models():
    model_holder.unload_models()


# TODO: code borrowed from the internet to be marked as such and to reside in separate files

def batched_background_removal(inimages, model_name):
    from rembg import new_session, remove
    print('creating background masks')
    outimages = []

    # model path and name
    bg_model_dir = Path.joinpath(Path().resolve(), "models/rem_bg")
    os.makedirs(bg_model_dir, exist_ok=True)
    os.environ["U2NET_HOME"] = str(bg_model_dir)

    # starting a session
    background_removal_session = new_session(model_name)
    for count in range(0, len(inimages)):
        bg_remove_img = np.array(remove(inimages[count], session=background_removal_session))
        outimages.append(Image.fromarray(bg_remove_img))
    # The line below might be redundant
    del background_removal_session
    return outimages


def pano_depth_to_world_points(depth):
    """
    360 depth to world points
    given 2D depth is an equirectangular projection of a spherical image
    Treat depth as radius
    longitude : -pi to pi
    latitude : -pi/2 to pi/2
    """

    # Convert depth to radius
    radius = depth.flatten()

    lon = np.linspace(-np.pi, np.pi, depth.shape[1])
    lat = np.linspace(-np.pi / 2, np.pi / 2, depth.shape[0])

    lon, lat = np.meshgrid(lon, lat)
    lon = lon.flatten()
    lat = lat.flatten()

    # Convert to cartesian coordinates
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)

    pts3d = np.stack([x, y, z], axis=1)

    return pts3d


def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask


def create_mesh(image, depth, keep_edges=False, spherical=False):
    import trimesh
    from dzoedepth.utils.geometry import depth_to_points, create_triangles
    maxsize = backbone.get_opt('depthmap_script_mesh_maxsize', 2048)

    # limit the size of the input image
    image.thumbnail((maxsize, maxsize))

    if not spherical:
        pts3d = depth_to_points(depth[None])
    else:
        pts3d = pano_depth_to_world_points(depth)

    pts3d = pts3d.reshape(-1, 3)

    verts = pts3d.reshape(-1, 3)
    image = np.array(image)
    if keep_edges:
        triangles = create_triangles(image.shape[0], image.shape[1])
    else:
        triangles = create_triangles(image.shape[0], image.shape[1], mask=~depth_edges_mask(depth))
    colors = image.reshape(-1, 3)

    mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

    # rotate 90deg over X when spherical
    if spherical:
        angle = math.pi / 2
        direction = [1, 0, 0]
        center = [0, 0, 0]
        rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
        mesh.apply_transform(rot_matrix)

    return mesh
