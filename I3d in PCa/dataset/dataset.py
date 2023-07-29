# -*- coding = utf-8 -*-
# @Time : 2022/5/27 17:40
# @Author : Tetsuya Chen
# @File : dataset.py
# @software : PyCharm
import os

import PIL
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as trans
from albumentations import *
from albumentations import pytorch as AT



train_transform = Compose([
    OneOf([
            GaussNoise(),
            ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.3),], p=0.3),

    OneOf([
            MotionBlur(p=0.3),
            MedianBlur(blur_limit=3, p=0.3),
            Blur(blur_limit=3, p=0.3),
            GaussianBlur(blur_limit=3, always_apply=False, p=0.3)
        ], p=0.5),
    # OneOf([
    #         # 畸变相关操作
    #         OpticalDistortion(p=0.2),
    #         GridDistortion(p=.2),
    #         #PiecewiseAffine(scale=(0, 0.05), p=0.3),
    #     ], p=0.5),
    # OneOf([
    #         # 锐化、浮雕等操作
    #         CLAHE(clip_limit=2),
    #         Sharpen(),
    #         Emboss(),
    #         RandomBrightnessContrast(),
    #     ], p=0.5),
    RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0),
    #HueSaturationValue(p=0.3),
    #RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
    #ChannelShuffle(always_apply=False, p=0.5),
    #augmentations.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, always_apply=False, p=0.5),
    augmentations.geometric.rotate.SafeRotate(limit=20, interpolation=cv2.INTER_CUBIC, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
    CropAndPad((-10, 10), p=0.5),
    RandomRotate90(p=0.5),
    Transpose(p=0.5),
    Flip(p=0.5),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0625, rotate_limit=45, border_mode=1, p=0.5, interpolation=cv2.INTER_CUBIC),
    RandomResizedCrop(224, 224, scale=(0.8, 1.0), ratio=(1, 1), interpolation=cv2.INTER_CUBIC, always_apply=False, p=0.5),
    #CoarseDropout (max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
    #GridDropout (ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, random_offset=False, fill_value=0, mask_fill_value=None,p=0.5),
    # to fit moblie_vit
    augmentations.geometric.resize.Resize(256, 256, interpolation=1, always_apply=False, p=1),

    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    AT.ToTensorV2(),
])

test_transform = Compose([
    # to fit moblie_vit
    augmentations.geometric.resize.Resize(256, 256, interpolation=1, always_apply=False, p=1),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    AT.ToTensorV2(),
])

stack_train_transform = Compose([
    # to fit moblie_vit

    OneOf([
            GaussNoise(p=0.3),
            #MotionBlur(p=0.3),
            #MedianBlur(blur_limit=3, p=0.3),
            Blur(blur_limit=3, p=0.3),
            GaussianBlur(blur_limit=3, always_apply=False, p=0.3)
        ], p=0.5),
    RandomRotate90(p=0.5),
    Transpose(p=0.5),
    Flip(p=0.5),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0625, rotate_limit=45, border_mode=1, p=0.5, interpolation=cv2.INTER_CUBIC),
    RandomResizedCrop(256, 256, scale=(0.8, 1.0), ratio=(1, 1), interpolation=cv2.INTER_CUBIC, always_apply=False, p=0.5),
    CoarseDropout (max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
    GridDropout (ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, random_offset=False, fill_value=0, mask_fill_value=None,p=0.5),
    augmentations.geometric.resize.Resize(256, 256, interpolation=1, always_apply=False, p=1),
    Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0, p=1.0),

    AT.ToTensorV2(),
])

stack_test_transform = Compose([
    # to fit moblie_vit
    augmentations.geometric.resize.Resize(256, 256, interpolation=1, always_apply=False, p=1),
    Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0, p=1.0),
    AT.ToTensorV2(),
])

class TrainDataset(Dataset):

    def __init__(self, df, transform=train_transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path, label = self.df.path[idx], self.df.label[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']

        return img, label

class TestDataset(Dataset):

    def __init__(self, df, transform=test_transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path, label = self.df.path[idx], self.df.label[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        return img, label

# class TrainDataset_random(Dataset):
#
#     def __init__(self, df, range=False, transform=train_transform):
#         self.df = df
#         self.transform = transform
#         self.range = range
#
#     def __len__(self):
#         return self.df.shape[0]
#
#     def __getitem__(self, idx):
#         if self.range:
#             min_idx = int(self.range * )
#         img_path, label = self.df.path[idx], self.df.label[idx]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = self.transform(image=img)['image']
#         return img, label

class heatmap_dataset(Dataset):
    def __init__(self, df, transform=test_transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path, label, num = self.df.path[idx], self.df.label[idx], self.df.id[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_ = self.transform(image=img)['image']
        return img, img_, label, num

class split_patch_dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.df.path[idx]
        img_folder = img_path[:img_path.rfind('/')]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, img_folder

class stack_train_dataset(Dataset):
    def __init__(self, df, transform=stack_train_transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        seed = np.random.randint(0, 100)
        img_dir, label = self.df.path[idx], self.df.label[idx]
        img_nums = []
        idx = 0
        imgs = torch.Tensor([0])
        for img_path in os.listdir(img_dir):
            img_nums.append(int((img_path.split('img')[1]).split('.')[0]))
        img_nums = sorted(img_nums)
        for img_num in img_nums:
            np.random.seed(seed)
            img_path = os.path.join(img_dir, f'img{img_num}.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = self.transform(image=img)['image']

            if idx == 0:
                imgs = img

            else:
                imgs = torch.cat((imgs, img), 0)
            idx += 1

        return imgs, label

class stack_test_dataset(Dataset):
    def __init__(self, df, transform=stack_test_transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_dir, label = self.df.path[idx], self.df.label[idx]
        img_nums = []
        idx = 0
        imgs = torch.Tensor([0])
        for img_path in os.listdir(img_dir):
            img_nums.append(int((img_path.split('img')[1]).split('.')[0]))
        img_nums = sorted(img_nums)
        for img_num in img_nums:
            img_path = os.path.join(img_dir, f'img{img_num}.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = self.transform(image=img)['image']
            if idx == 0:
                imgs = img

            else:
                imgs = torch.cat((imgs, img), 0)
            idx += 1
        return imgs, label