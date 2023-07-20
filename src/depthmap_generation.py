import gc
import os.path
from operator import getitem

import cv2
import numpy as np
import skimage.measure
from PIL import Image
import torch
from torchvision.transforms import Compose, transforms

# midas imports
from dmidas.dpt_depth import DPTDepthModel
from dmidas.midas_net import MidasNet
from dmidas.midas_net_custom import MidasNet_small
from dmidas.transforms import Resize, NormalizeImage, PrepareForNet
# zoedepth
from dzoedepth.models.builder import build_model
from dzoedepth.utils.config import get_config
# AdelaiDepth/LeReS imports
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import strip_prefix_if_present
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
# pix2pix/merge net imports
from pix2pix.options.test_options import TestOptions

# Our code
from src.misc import *
from src import backbone

global depthmap_device

class ModelHolder:
    def __init__(self):
        self.depth_model = None
        self.pix2pix_model = None
        self.depth_model_type = None
        self.device = None  # Target device, the model may be swapped from VRAM into RAM.
        self.offloaded = False  # True means current device is not the target device

        # Extra stuff
        self.resize_mode = None
        self.normalization = None

        # Settings (initialized to sensible values, should be updated)
        self.boost_whole_size_threshold = 1600  # R_max from the paper by default
        self.no_half = False
        self.precision = "autocast"

    def update_settings(self, boost_whole_size_threshold=None, no_half=None, precision=None):
        if boost_whole_size_threshold is not None:
            self.boost_whole_size_threshold = boost_whole_size_threshold
        if no_half is not None:
            self.no_half = no_half
        if precision is not None:
            self.precision = precision


    def ensure_models(self, model_type, device: torch.device, boost: bool):
        # TODO: could make it more granular
        if model_type == -1 or model_type is None:
            self.unload_models()
            return
        # Certain optimisations are irreversible and not device-agnostic, thus changing device requires reloading
        if model_type != self.depth_model_type or boost != (self.pix2pix_model is not None) or device != self.device:
            self.unload_models()
            self.load_models(model_type, device, boost)
        self.reload()

    def load_models(self, model_type, device: torch.device, boost: bool):
        """Ensure that the depth model is loaded"""

        # model path and name
        model_dir = "./models/midas"
        if model_type == 0:
            model_dir = "./models/leres"
        # create paths to model if not present
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs('./models/pix2pix', exist_ok=True)

        print("Loading model weights from ", end=" ")

        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        model = None
        if model_type == 0:  # "res101"
            model_path = f"{model_dir}/res101.pth"
            print(model_path)
            ensure_file_downloaded(
                model_path,
                ["https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download",
                 "https://huggingface.co/lllyasviel/Annotators/resolve/5bc80eec2b4fddbb/res101.pth",
                 ],
                "1d696b2ef3e8336b057d0c15bc82d2fecef821bfebe5ef9d7671a5ec5dde520b")
            if device != torch.device('cpu'):
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model = RelDepthModel(backbone='resnext101')
            model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."), strict=True)
            del checkpoint
            backbone.torch_gc()

        if model_type == 1:  # "dpt_beit_large_512" midas 3.1
            model_path = f"{model_dir}/dpt_beit_large_512.pt"
            print(model_path)
            ensure_file_downloaded(model_path,
                                   "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt")
            model = DPTDepthModel(
                path=model_path,
                backbone="beitl16_512",
                non_negative=True,
            )
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if model_type == 2:  # "dpt_beit_large_384" midas 3.1
            model_path = f"{model_dir}/dpt_beit_large_384.pt"
            print(model_path)
            ensure_file_downloaded(model_path,
                                   "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt")
            model = DPTDepthModel(
                path=model_path,
                backbone="beitl16_384",
                non_negative=True,
            )
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if model_type == 3:  # "dpt_large_384" midas 3.0
            model_path = f"{model_dir}/dpt_large-midas-2f21e586.pt"
            print(model_path)
            ensure_file_downloaded(model_path,
                                   "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt")
            model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
            )
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == 4:  # "dpt_hybrid_384" midas 3.0
            model_path = f"{model_dir}/dpt_hybrid-midas-501f0c75.pt"
            print(model_path)
            ensure_file_downloaded(model_path,
                                   "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt")
            model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == 5:  # "midas_v21"
            model_path = f"{model_dir}/midas_v21-f6b98070.pt"
            print(model_path)
            ensure_file_downloaded(model_path,
                                   "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt")
            model = MidasNet(model_path, non_negative=True)
            resize_mode = "upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        elif model_type == 6:  # "midas_v21_small"
            model_path = f"{model_dir}/midas_v21_small-70d6b9c8.pt"
            print(model_path)
            ensure_file_downloaded(model_path,
                                   "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt")
            model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                                   non_negative=True, blocks={'expand': True})
            resize_mode = "upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        # When loading, zoedepth models will report the default net size.
        # It will be overridden by the generation settings.
        elif model_type == 7:  # zoedepth_n
            print("zoedepth_n\n")
            conf = get_config("zoedepth", "infer")
            model = build_model(conf)

        elif model_type == 8:  # zoedepth_k
            print("zoedepth_k\n")
            conf = get_config("zoedepth", "infer", config_version="kitti")
            model = build_model(conf)

        elif model_type == 9:  # zoedepth_nk
            print("zoedepth_nk\n")
            conf = get_config("zoedepth_nk", "infer")
            model = build_model(conf)

        model.eval()  # prepare for evaluation
        # optimize
        if device == torch.device("cuda") and model_type in [0, 1, 2, 3, 4, 5, 6]:
            model = model.to(memory_format=torch.channels_last)  # TODO: weird
            if not self.no_half and model_type != 0 and not boost:  # TODO: zoedepth, too?
                model = model.half()
        model.to(device)  # to correct device

        self.depth_model = model
        self.depth_model_type = model_type
        self.resize_mode = resize_mode
        self.normalization = normalization

        self.device = device

        if boost:
            # sfu.ca unfortunately is not very reliable, we use a mirror just in case
            ensure_file_downloaded(
                './models/pix2pix/latest_net_G.pth',
                ["https://huggingface.co/lllyasviel/Annotators/resolve/9a7d84251d487d11/latest_net_G.pth",
                 "https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth"],
                '50ec735d74ed6499562d898f41b49343e521808b8dae589aa3c2f5c9ac9f7462')
            opt = TestOptions().parse()
            if device == torch.device('cpu'):
                opt.gpu_ids = []
            self.pix2pix_model = Pix2Pix4DepthModel(opt)
            self.pix2pix_model.save_dir = './models/pix2pix'
            self.pix2pix_model.load_networks('latest')
            self.pix2pix_model.eval()

        backbone.torch_gc()

    @staticmethod
    def get_default_net_size(model_type):
        sizes = {
            0: [448, 448],
            1: [512, 512],
            2: [384, 384],
            3: [384, 384],
            4: [384, 384],
            5: [384, 384],
            6: [256, 256],
            7: [384, 512],
            8: [384, 768],
            9: [384, 512]
        }
        if model_type in sizes:
            return sizes[model_type]
        return [512, 512]

    def offload(self):
        """Move to RAM to conserve VRAM"""
        if self.device != torch.device('cpu') and not self.offloaded:
            self.move_models_to(torch.device('cpu'))
            self.offloaded = True

    def reload(self):
        """Undoes offload"""
        if self.offloaded:
            self.move_models_to(self.device)
            self.offloaded = False

    def move_models_to(self, device):
        if self.depth_model is not None:
            self.depth_model.to(device)
        if self.pix2pix_model is not None:
            pass
            # TODO: pix2pix offloading not implemented

    def unload_models(self):
        if self.depth_model is not None or self.pix2pix_model is not None:
            del self.depth_model
            self.depth_model = None
            del self.pix2pix_model
            self.pix2pix_model = None
            gc.collect()
            backbone.torch_gc()

        self.depth_model_type = None
        self.device = None

    def get_raw_prediction(self, input, net_width, net_height):
        """Get prediction from the model currently loaded by the ModelHolder object.
        If boost is enabled, net_width and net_height will be ignored."""
        global depthmap_device
        depthmap_device = self.device
        # input image
        img = cv2.cvtColor(np.asarray(input), cv2.COLOR_BGR2RGB) / 255.0
        # compute depthmap
        if self.pix2pix_model is None:
            if self.depth_model_type == 0:
                raw_prediction = estimateleres(img, self.depth_model, net_width, net_height)
            elif self.depth_model_type in [7, 8, 9]:
                raw_prediction = estimatezoedepth(input, self.depth_model, net_width, net_height)
            else:
                raw_prediction = estimatemidas(img, self.depth_model, net_width, net_height,
                                               self.resize_mode, self.normalization, self.no_half,
                                               self.precision == "autocast")
        else:
            raw_prediction = estimateboost(img, self.depth_model, self.depth_model_type, self.pix2pix_model,
                                           self.boost_whole_size_threshold)
        raw_prediction_invert = self.depth_model_type in [0, 7, 8, 9]
        return raw_prediction, raw_prediction_invert


def estimateleres(img, model, w, h):
    # leres transform input
    rgb_c = img[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (w, h))
    img_torch = scale_torch(A_resize)[None, :, :, :]

    # compute
    with torch.no_grad():
        if depthmap_device == torch.device("cuda"):
            img_torch = img_torch.cuda()
        prediction = model.depth_model(img_torch)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return prediction


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = transform(img.astype(np.float32))
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


def estimatezoedepth(img, model, w, h):
    # x = transforms.ToTensor()(img).unsqueeze(0)
    # x = x.type(torch.float32)
    # x.to(depthmap_device)
    # prediction = model.infer(x)
    model.core.prep.resizer._Resize__width = w
    model.core.prep.resizer._Resize__height = h
    prediction = model.infer_pil(img)

    return prediction


def estimatemidas(img, model, w, h, resize_mode, normalization, no_half, precision_is_autocast):
    import contextlib
    # init transform
    transform = Compose(
        [
            Resize(
                w,
                h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    # transform input
    img_input = transform({"image": img})["image"]

    # compute
    precision_scope = torch.autocast if precision_is_autocast and depthmap_device == torch.device(
        "cuda") else contextlib.nullcontext
    with torch.no_grad(), precision_scope("cuda"):
        sample = torch.from_numpy(img_input).to(depthmap_device).unsqueeze(0)
        if depthmap_device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            if not no_half:
                sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


class ImageandPatchs:
    def __init__(self, root_dir, name, patchsinfo, rgb_image, scale=1):
        self.root_dir = root_dir
        self.patchsinfo = patchsinfo
        self.name = name
        self.patchs = patchsinfo
        self.scale = scale

        self.rgb_image = cv2.resize(rgb_image, (round(rgb_image.shape[1] * scale), round(rgb_image.shape[0] * scale)),
                                    interpolation=cv2.INTER_CUBIC)

        self.do_have_estimate = False
        self.estimation_updated_image = None
        self.estimation_base_image = None

    def __len__(self):
        return len(self.patchs)

    def set_base_estimate(self, est):
        self.estimation_base_image = est
        if self.estimation_updated_image is not None:
            self.do_have_estimate = True

    def set_updated_estimate(self, est):
        self.estimation_updated_image = est
        if self.estimation_base_image is not None:
            self.do_have_estimate = True

    def __getitem__(self, index):
        patch_id = int(self.patchs[index][0])
        rect = np.array(self.patchs[index][1]['rect'])
        msize = self.patchs[index][1]['size']

        ## applying scale to rect:
        rect = np.round(rect * self.scale)
        rect = rect.astype('int')
        msize = round(msize * self.scale)

        patch_rgb = impatch(self.rgb_image, rect)
        if self.do_have_estimate:
            patch_whole_estimate_base = impatch(self.estimation_base_image, rect)
            patch_whole_estimate_updated = impatch(self.estimation_updated_image, rect)
            return {'patch_rgb': patch_rgb, 'patch_whole_estimate_base': patch_whole_estimate_base,
                    'patch_whole_estimate_updated': patch_whole_estimate_updated, 'rect': rect,
                    'size': msize, 'id': patch_id}
        else:
            return {'patch_rgb': patch_rgb, 'rect': rect, 'size': msize, 'id': patch_id}

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        """
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        """

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #    torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


def impatch(image, rect):
    # Extract the given patch pixels from a given image.
    w1 = rect[0]
    h1 = rect[1]
    w2 = w1 + rect[2]
    h2 = h1 + rect[3]
    image_patch = image[h1:h2, w1:w2]
    return image_patch


class ImageandPatchs:
    def __init__(self, root_dir, name, patchsinfo, rgb_image, scale=1):
        self.root_dir = root_dir
        self.patchsinfo = patchsinfo
        self.name = name
        self.patchs = patchsinfo
        self.scale = scale

        self.rgb_image = cv2.resize(rgb_image, (round(rgb_image.shape[1] * scale), round(rgb_image.shape[0] * scale)),
                                    interpolation=cv2.INTER_CUBIC)

        self.do_have_estimate = False
        self.estimation_updated_image = None
        self.estimation_base_image = None

    def __len__(self):
        return len(self.patchs)

    def set_base_estimate(self, est):
        self.estimation_base_image = est
        if self.estimation_updated_image is not None:
            self.do_have_estimate = True

    def set_updated_estimate(self, est):
        self.estimation_updated_image = est
        if self.estimation_base_image is not None:
            self.do_have_estimate = True

    def __getitem__(self, index):
        patch_id = int(self.patchs[index][0])
        rect = np.array(self.patchs[index][1]['rect'])
        msize = self.patchs[index][1]['size']

        ## applying scale to rect:
        rect = np.round(rect * self.scale)
        rect = rect.astype('int')
        msize = round(msize * self.scale)

        patch_rgb = impatch(self.rgb_image, rect)
        if self.do_have_estimate:
            patch_whole_estimate_base = impatch(self.estimation_base_image, rect)
            patch_whole_estimate_updated = impatch(self.estimation_updated_image, rect)
            return {'patch_rgb': patch_rgb, 'patch_whole_estimate_base': patch_whole_estimate_base,
                    'patch_whole_estimate_updated': patch_whole_estimate_updated, 'rect': rect,
                    'size': msize, 'id': patch_id}
        else:
            return {'patch_rgb': patch_rgb, 'rect': rect, 'size': msize, 'id': patch_id}

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        """
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        """

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #    torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


def estimateboost(img, model, model_type, pix2pixmodel, whole_size_threshold):
    pix2pixsize = 1024  # TODO: pix2pixsize and whole_size_threshold to setting?

    if model_type == 0:  # leres
        net_receptive_field_size = 448
        patch_netsize = 2 * net_receptive_field_size
    elif model_type == 1:  # dpt_beit_large_512
        net_receptive_field_size = 512
        patch_netsize = 2 * net_receptive_field_size
    else:  # other midas
        net_receptive_field_size = 384
        patch_netsize = 2 * net_receptive_field_size

    gc.collect()
    backbone.torch_gc()

    # Generate mask used to smoothly blend the local pathc estimations to the base estimate.
    # It is arbitrarily large to avoid artifacts during rescaling for each crop.
    mask_org = generatemask((3000, 3000))
    mask = mask_org.copy()

    # Value x of R_x defined in the section 5 of the main paper.
    r_threshold_value = 0.2
    # if R0:
    #    r_threshold_value = 0

    input_resolution = img.shape
    scale_threshold = 3  # Allows up-scaling with a scale up to 3

    # Find the best input resolution R-x. The resolution search described in section 5-double estimation of the main paper and section B of the
    # supplementary material.
    whole_image_optimal_size, patch_scale = calculateprocessingres(img, net_receptive_field_size, r_threshold_value,
                                                                   scale_threshold, whole_size_threshold)

    print('wholeImage being processed in :', whole_image_optimal_size)

    # Generate the base estimate using the double estimation.
    whole_estimate = doubleestimate(img, net_receptive_field_size, whole_image_optimal_size, pix2pixsize, model,
                                    model_type, pix2pixmodel)

    # Compute the multiplier described in section 6 of the main paper to make sure our initial patch can select
    # small high-density regions of the image.
    factor = max(min(1, 4 * patch_scale * whole_image_optimal_size / whole_size_threshold), 0.2)
    print('Adjust factor is:', 1 / factor)

    # Compute the default target resolution.
    if img.shape[0] > img.shape[1]:
        a = 2 * whole_image_optimal_size
        b = round(2 * whole_image_optimal_size * img.shape[1] / img.shape[0])
    else:
        a = round(2 * whole_image_optimal_size * img.shape[0] / img.shape[1])
        b = 2 * whole_image_optimal_size
    b = int(round(b / factor))
    a = int(round(a / factor))

    """
    # recompute a, b and saturate to max res.
    if max(a,b) > max_res:
        print('Default Res is higher than max-res: Reducing final resolution')
        if img.shape[0] > img.shape[1]:
            a = max_res
            b = round(option.max_res * img.shape[1] / img.shape[0])
        else:
            a = round(option.max_res * img.shape[0] / img.shape[1])
            b = max_res
        b = int(b)
        a = int(a)
    """

    img = cv2.resize(img, (b, a), interpolation=cv2.INTER_CUBIC)

    # Extract selected patches for local refinement
    base_size = net_receptive_field_size * 2
    patchset = generatepatchs(img, base_size, factor)

    print('Target resolution: ', img.shape)

    # Computing a scale in case user prompted to generate the results as the same resolution of the input.
    # Notice that our method output resolution is independent of the input resolution and this parameter will only
    # enable a scaling operation during the local patch merge implementation to generate results with the same resolution
    # as the input.
    """
    if output_resolution == 1:
        mergein_scale = input_resolution[0] / img.shape[0]
        print('Dynamicly change merged-in resolution; scale:', mergein_scale)
    else:
        mergein_scale = 1
    """
    # always rescale to input res for now
    mergein_scale = input_resolution[0] / img.shape[0]

    imageandpatchs = ImageandPatchs('', '', patchset, img, mergein_scale)
    whole_estimate_resized = cv2.resize(whole_estimate, (round(img.shape[1] * mergein_scale),
                                                         round(img.shape[0] * mergein_scale)),
                                        interpolation=cv2.INTER_CUBIC)
    imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
    imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())

    print('Resulting depthmap resolution will be :', whole_estimate_resized.shape[:2])
    print('patches to process: ' + str(len(imageandpatchs)))

    # Enumerate through all patches, generate their estimations and refining the base estimate.
    for patch_ind in range(len(imageandpatchs)):

        # Get patch information
        patch = imageandpatchs[patch_ind]  # patch object
        patch_rgb = patch['patch_rgb']  # rgb patch
        patch_whole_estimate_base = patch['patch_whole_estimate_base']  # corresponding patch from base
        rect = patch['rect']  # patch size and location
        patch_id = patch['id']  # patch ID
        org_size = patch_whole_estimate_base.shape  # the original size from the unscaled input
        print('\t processing patch', patch_ind, '/', len(imageandpatchs) - 1, '|', rect)

        # We apply double estimation for patches. The high resolution value is fixed to twice the receptive
        # field size of the network for patches to accelerate the process.
        patch_estimation = doubleestimate(patch_rgb, net_receptive_field_size, patch_netsize, pix2pixsize, model,
                                          model_type, pix2pixmodel)
        patch_estimation = cv2.resize(patch_estimation, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)
        patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (pix2pixsize, pix2pixsize),
                                               interpolation=cv2.INTER_CUBIC)

        # Merging the patch estimation into the base estimate using our merge network:
        # We feed the patch estimation and the same region from the updated base estimate to the merge network
        # to generate the target estimate for the corresponding region.
        pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)

        # Run merging network
        pix2pixmodel.test()
        visuals = pix2pixmodel.get_current_visuals()

        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped + 1) / 2
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

        mapped = prediction_mapped

        # We use a simple linear polynomial to make sure the result of the merge network would match the values of
        # base estimate
        p_coef = np.polyfit(mapped.reshape(-1), patch_whole_estimate_base.reshape(-1), deg=1)
        merged = np.polyval(p_coef, mapped.reshape(-1)).reshape(mapped.shape)

        merged = cv2.resize(merged, (org_size[1], org_size[0]), interpolation=cv2.INTER_CUBIC)

        # Get patch size and location
        w1 = rect[0]
        h1 = rect[1]
        w2 = w1 + rect[2]
        h2 = h1 + rect[3]

        # To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
        # and resize it to our needed size while merging the patches.
        if mask.shape != org_size:
            mask = cv2.resize(mask_org, (org_size[1], org_size[0]), interpolation=cv2.INTER_LINEAR)

        tobemergedto = imageandpatchs.estimation_updated_image

        # Update the whole estimation:
        # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
        # blending at the boundaries of the patch region.
        tobemergedto[h1:h2, w1:w2] = np.multiply(tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
        imageandpatchs.set_updated_estimate(tobemergedto)

    # output
    return cv2.resize(imageandpatchs.estimation_updated_image, (input_resolution[1], input_resolution[0]),
                      interpolation=cv2.INTER_CUBIC)


def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0] / 16)
    k_size = int(2 * np.ceil(2 * int(size[0] / 16)) + 1)
    mask[int(0.15 * size[0]):size[0] - int(0.15 * size[0]), int(0.15 * size[1]): size[1] - int(0.15 * size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask


def rgb2gray(rgb):
    # Converts rgb to gray
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def resizewithpool(img, size):
    i_size = img.shape[0]
    n = int(np.floor(i_size / size))

    out = skimage.measure.block_reduce(img, (n, n), np.max)
    return out


def calculateprocessingres(img, basesize, confidence=0.1, scale_threshold=3, whole_size_threshold=3000):
    # Returns the R_x resolution described in section 5 of the main paper.

    # Parameters:
    #    img :input rgb image
    #    basesize : size the dilation kernel which is equal to receptive field of the network.
    #    confidence: value of x in R_x; allowed percentage of pixels that are not getting any contextual cue.
    #    scale_threshold: maximum allowed upscaling on the input image ; it has been set to 3.
    #    whole_size_threshold: maximum allowed resolution. (R_max from section 6 of the main paper)

    # Returns:
    #    outputsize_scale*speed_scale :The computed R_x resolution
    #    patch_scale: K parameter from section 6 of the paper

    # speed scale parameter is to process every image in a smaller size to accelerate the R_x resolution search
    speed_scale = 32
    image_dim = int(min(img.shape[0:2]))

    gray = rgb2gray(img)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad = cv2.resize(grad, (image_dim, image_dim), cv2.INTER_AREA)

    # thresholding the gradient map to generate the edge-map as a proxy of the contextual cues
    m = grad.min()
    M = grad.max()
    middle = m + (0.4 * (M - m))
    grad[grad < middle] = 0
    grad[grad >= middle] = 1

    # dilation kernel with size of the receptive field
    kernel = np.ones((int(basesize / speed_scale), int(basesize / speed_scale)), float)
    # dilation kernel with size of the a quarter of receptive field used to compute k
    # as described in section 6 of main paper
    kernel2 = np.ones((int(basesize / (4 * speed_scale)), int(basesize / (4 * speed_scale))), float)

    # Output resolution limit set by the whole_size_threshold and scale_threshold.
    threshold = min(whole_size_threshold, scale_threshold * max(img.shape[:2]))

    outputsize_scale = basesize / speed_scale
    for p_size in range(int(basesize / speed_scale), int(threshold / speed_scale), int(basesize / (2 * speed_scale))):
        grad_resized = resizewithpool(grad, p_size)
        grad_resized = cv2.resize(grad_resized, (p_size, p_size), cv2.INTER_NEAREST)
        grad_resized[grad_resized >= 0.5] = 1
        grad_resized[grad_resized < 0.5] = 0

        dilated = cv2.dilate(grad_resized, kernel, iterations=1)
        meanvalue = (1 - dilated).mean()
        if meanvalue > confidence:
            break
        else:
            outputsize_scale = p_size

    grad_region = cv2.dilate(grad_resized, kernel2, iterations=1)
    patch_scale = grad_region.mean()

    return int(outputsize_scale * speed_scale), patch_scale


# Generate a double-input depth estimation
def doubleestimate(img, size1, size2, pix2pixsize, model, net_type, pix2pixmodel):
    # Generate the low resolution estimation
    estimate1 = singleestimate(img, size1, model, net_type)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = singleestimate(img, size2, model, net_type)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped + 1) / 2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
            torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


# Generate a single-input depth estimation
def singleestimate(img, msize, model, net_type):
    if net_type == 0:
        return estimateleres(img, model, msize, msize)
    elif net_type >= 7:
        # np to PIL
        return estimatezoedepth(Image.fromarray(np.uint8(img * 255)).convert('RGB'), model, msize, msize)
    else:
        return estimatemidasBoost(img, model, msize, msize)


# Generating local patches to perform the local refinement described in section 6 of the main paper.
def generatepatchs(img, base_size, factor):
    # Compute the gradients as a proxy of the contextual cues.
    img_gray = rgb2gray(img)
    whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) + \
                 np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0

    # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
    gf = whole_grad.sum() / len(whole_grad.reshape(-1))
    grad_integral_image = cv2.integral(whole_grad)

    # Variables are selected such that the initial patch size would be the receptive field size
    # and the stride is set to 1/3 of the receptive field size.
    blsize = int(round(base_size / 2))
    stride = int(round(blsize * 0.75))

    # Get initial Grid
    patch_bound_list = applyGridpatch(blsize, stride, img, [0, 0, 0, 0])

    # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
    # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
    print("Selecting patches ...")
    patch_bound_list = adaptiveselection(grad_integral_image, patch_bound_list, gf, factor)

    # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
    # patch
    patchset = sorted(patch_bound_list.items(), key=lambda x: getitem(x[1], 'size'), reverse=True)
    return patchset


def applyGridpatch(blsize, stride, img, box):
    # Extract a simple grid patch.
    counter1 = 0
    patch_bound_list = {}
    for k in range(blsize, img.shape[1] - blsize, stride):
        for j in range(blsize, img.shape[0] - blsize, stride):
            patch_bound_list[str(counter1)] = {}
            patchbounds = [j - blsize, k - blsize, j - blsize + 2 * blsize, k - blsize + 2 * blsize]
            patch_bound = [box[0] + patchbounds[1], box[1] + patchbounds[0], patchbounds[3] - patchbounds[1],
                           patchbounds[2] - patchbounds[0]]
            patch_bound_list[str(counter1)]['rect'] = patch_bound
            patch_bound_list[str(counter1)]['size'] = patch_bound[2]
            counter1 = counter1 + 1
    return patch_bound_list


# Adaptively select patches
def adaptiveselection(integral_grad, patch_bound_list, gf, factor):
    patchlist = {}
    count = 0
    height, width = integral_grad.shape

    search_step = int(32 / factor)

    # Go through all patches
    for c in range(len(patch_bound_list)):
        # Get patch
        bbox = patch_bound_list[str(c)]['rect']

        # Compute the amount of gradients present in the patch from the integral image.
        cgf = getGF_fromintegral(integral_grad, bbox) / (bbox[2] * bbox[3])

        # Check if patching is beneficial by comparing the gradient density of the patch to
        # the gradient density of the whole image
        if cgf >= gf:
            bbox_test = bbox.copy()
            patchlist[str(count)] = {}

            # Enlarge each patch until the gradient density of the patch is equal
            # to the whole image gradient density
            while True:

                bbox_test[0] = bbox_test[0] - int(search_step / 2)
                bbox_test[1] = bbox_test[1] - int(search_step / 2)

                bbox_test[2] = bbox_test[2] + search_step
                bbox_test[3] = bbox_test[3] + search_step

                # Check if we are still within the image
                if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                        or bbox_test[0] + bbox_test[2] >= width:
                    break

                # Compare gradient density
                cgf = getGF_fromintegral(integral_grad, bbox_test) / (bbox_test[2] * bbox_test[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()

            # Add patch to selected patches
            patchlist[str(count)]['rect'] = bbox
            patchlist[str(count)]['size'] = bbox[2]
            count = count + 1

    # Return selected patches
    return patchlist


def getGF_fromintegral(integralimage, rect):
    # Computes the gradient density of a given patch from the gradient integral image.
    x1 = rect[1]
    x2 = rect[1] + rect[3]
    y1 = rect[0]
    y2 = rect[0] + rect[2]
    value = integralimage[x2, y2] - integralimage[x1, y2] - integralimage[x2, y1] + integralimage[x1, y1]
    return value


def estimatemidasBoost(img, model, w, h):
    # init transform
    transform = Compose(
        [
            Resize(
                w,
                h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    # transform input
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(depthmap_device).unsqueeze(0)
        if depthmap_device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
        prediction = model.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction
