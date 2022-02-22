# Copyright (c) 2022-present, Js2hou.
# All rights reserved.

# Implements dataset here.

from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


def build_dataset(data_path='data/cifar100'):
    train_dataset = datasets.CIFAR100(
        data_path, train=True, download=True, transform=build_transform(True))
    val_dataset = datasets.CIFAR100(
        data_path, train=False, download=True, transform=build_transform(False))
    nb_classes = 100
    return train_dataset, val_dataset, nb_classes


def build_transform(is_train):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size),
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
