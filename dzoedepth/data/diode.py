# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x : x
        self.resize = transforms.Resize(480)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        image = self.resize(image)

        return {'image': image, 'depth': depth, 'dataset': "diode"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class DIODE(Dataset):
    def __init__(self, data_dir_root):
        import glob

        # image paths are of the form <data_dir_root>/scene_#/scan_#/*.png
        self.image_files = glob.glob(
            os.path.join(data_dir_root, '*', '*', '*.png'))
        self.depth_files = [r.replace(".png", "_depth.npy")
                            for r in self.image_files]
        self.depth_mask_files = [
            r.replace(".png", "_depth_mask.npy") for r in self.image_files]
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]
        depth_mask_path = self.depth_mask_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.load(depth_path)  # in meters
        valid = np.load(depth_mask_path)  # binary

        # depth[depth > 8] = -1
        # depth = depth[..., None]

        sample = dict(image=image, depth=depth, valid=valid)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)


def get_diode_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = DIODE(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)

# get_diode_loader(data_dir_root="datasets/diode/val/outdoor")
