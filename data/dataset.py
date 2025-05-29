import glob

import cv2
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
# from data.util.mask_generator_256_small import RandomMask
import os
import torch
import numpy as np

# # debug
# import matplotlib.pyplot as plt

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox, RandomMask, RandomMaskLarge, BatchRandomMask)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str_, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mask_path = self.mask_config['mask_path']

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        # 根据路径加载图像并进行处理
        img = self.tfs(self.loader(path))
        mask = self.get_mask(index)

        # # debug
        # plt.imshow(mask, cmap='gray')
        # plt.title('Mask Image')
        # plt.colorbar()
        # plt.show()

        #  被遮挡的区域被随机噪声替换，而不被遮挡的区域保持不变
        noise = torch.randn_like(img)
        cond_image = img*(1. - mask) + mask*noise
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        # ret['noise'] = noise
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, index):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'hybrid1':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = get_irregular_mask(self.image_size)
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'hybrid2':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            mat_large = 1 - RandomMaskLarge(self.image_size[0], hole_range=[0, 1]).transpose(1, 2, 0)
            mask = regular_mask | mat_large
        elif self.mask_mode == 'mat_small':
            mask = 1 - RandomMask(self.image_size[0], hole_range=[0, 1]).transpose(1, 2, 0)
        elif self.mask_mode == 'mat_large':
            mask = 1 - RandomMaskLarge(self.image_size[0], hole_range=[0, 1]).transpose(1, 2, 0)
        elif self.mask_mode == 'gen':
            mask = np.ones((self.image_size[0], self.image_size[1], 1), dtype=np.uint8)
        elif self.mask_mode == 'file':
            # debug
            # self.mask_path = 'datasets/celebahq_masks_256_small'
            if self.mask_path is None:
                self.mask_path = self.mask_config['mask_path']
                raise ValueError("mask_path must be specified when mask_mode is 'file'")
            mask_list = sorted(glob.glob(self.mask_path + '/*.png') + glob.glob(self.mask_path + '/*.jpg'))
            mask = cv2.imread(mask_list[index], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            # TODO:需要检查 mask 和 img 形状是否匹配
            if mask.shape != (self.image_size[0], self.image_size[1]):
                mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))
            mask = 1 - np.expand_dims(mask, axis=-1)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


