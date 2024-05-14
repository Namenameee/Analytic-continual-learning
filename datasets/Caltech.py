# -*- coding: utf-8 -*-

from typing import Tuple
import torch
from torchvision.datasets import Caltech256, Caltech101
from torchvision.transforms import v2 as transforms
from os import path
import shutil
import os
from .DatasetWrapper import DatasetWrapper


def remove_background_class(dataset_dir: str, background_dir_name: str):
    background_dir_path = path.join(dataset_dir, background_dir_name)
    if path.exists(background_dir_path):
        shutil.rmtree(background_dir_path)
        print(f"Removed background class directory: {background_dir_path}")


class Caltech256_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 256
    mean = (0.55197421, 0.53357298, 0.50502131)
    std = (0.31780326, 0.31446489, 0.32799368)

    basic_transform = transforms.Compose(
        [
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    augment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(176),
            transforms.RandomHorizontalFlip(0.5),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.RandomErasing(0.1),
            transforms.ToPureTensor(),
        ]
    )

    def __init__(
        self,
        root: str,
        train: bool,
        base_ratio: float,
        num_phases: int,
        augment: bool = False,
        inplace_repeat: int = 1,
        shuffle_seed: int | None = None,
    ) -> None:
        root = path.expanduser(root)
        remove_background_class(
            path.join(root, "caltech256/256_ObjectCategories"), "257.clutter"
        )
        self.dataset = Caltech256(root, download=True)
        super().__init__(
            [item[1] for item in self.dataset],
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )


class Caltech101_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 101
    mean = (0.54865471, 0.53127132, 0.50505934)
    std = (0.31725705, 0.31189067, 0.32411599)

    basic_transform = transforms.Compose(
        [
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    augment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(176),
            transforms.RandomHorizontalFlip(0.5),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.RandomErasing(0.1),
            transforms.ToPureTensor(),
        ]
    )

    def __init__(
        self,
        root: str,
        train: bool,
        base_ratio: float,
        num_phases: int,
        augment: bool = False,
        inplace_repeat: int = 1,
        shuffle_seed: int | None = None,
    ) -> None:
        root = path.expanduser(root)
        self.dataset = Caltech101(root, download=True)
        super().__init__(
            [item[1] for item in self.dataset],
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )
