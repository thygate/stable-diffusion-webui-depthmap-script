from pix2pix.data.base_dataset import BaseDataset
from pix2pix.data.image_folder import make_dataset
from pix2pix.util.guidedfilter import GuidedFilter

import numpy as np
import os
import torch
from PIL import Image


def normalize(img):
    img = img * 2
    img = img - 1
    return img


def normalize01(img):
    return (img - torch.min(img)) / (torch.max(img)-torch.min(img))


class DepthMergeDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_outer = os.path.join(opt.dataroot, opt.phase, 'outer')
        self.dir_inner = os.path.join(opt.dataroot, opt.phase, 'inner')
        self.dir_gtfake = os.path.join(opt.dataroot, opt.phase, 'gtfake')

        self.outer_paths = sorted(make_dataset(self.dir_outer, opt.max_dataset_size))
        self.inner_paths = sorted(make_dataset(self.dir_inner, opt.max_dataset_size))
        self.gtfake_paths = sorted(make_dataset(self.dir_gtfake, opt.max_dataset_size))

        self.dataset_size = len(self.outer_paths)

        if opt.phase == 'train':
            self.isTrain = True
        else:
            self.isTrain = False

    def __getitem__(self, index):
        normalize_coef = np.float32(2 ** 16)

        data_outer = Image.open(self.outer_paths[index % self.dataset_size])  # needs to be a tensor
        data_outer = np.array(data_outer, dtype=np.float32)
        data_outer = data_outer / normalize_coef

        data_inner = Image.open(self.inner_paths[index % self.dataset_size])  # needs to be a tensor
        data_inner = np.array(data_inner, dtype=np.float32)
        data_inner = data_inner / normalize_coef

        if self.isTrain:
            data_gtfake = Image.open(self.gtfake_paths[index % self.dataset_size])  # needs to be a tensor
            data_gtfake = np.array(data_gtfake, dtype=np.float32)
            data_gtfake = data_gtfake / normalize_coef

            data_inner = GuidedFilter(data_gtfake, data_inner, 64, 0.00000001).smooth.astype('float32')
            data_outer = GuidedFilter(data_outer, data_gtfake, 64, 0.00000001).smooth.astype('float32')

        data_outer = torch.from_numpy(data_outer)
        data_outer = torch.unsqueeze(data_outer, 0)
        data_outer = normalize01(data_outer)
        data_outer = normalize(data_outer)

        data_inner = torch.from_numpy(data_inner)
        data_inner = torch.unsqueeze(data_inner, 0)
        data_inner = normalize01(data_inner)
        data_inner = normalize(data_inner)

        if self.isTrain:
            data_gtfake = torch.from_numpy(data_gtfake)
            data_gtfake = torch.unsqueeze(data_gtfake, 0)
            data_gtfake = normalize01(data_gtfake)
            data_gtfake = normalize(data_gtfake)

        image_path = self.outer_paths[index % self.dataset_size]
        if self.isTrain:
            return {'data_inner': data_inner, 'data_outer': data_outer,
                    'data_gtfake': data_gtfake, 'image_path': image_path}
        else:
            return {'data_inner': data_inner, 'data_outer': data_outer, 'image_path': image_path}

    def __len__(self):
        """Return the total number of images."""
        return self.dataset_size
