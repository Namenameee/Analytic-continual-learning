# -*- coding: utf-8 -*-

import os
from os import path, mkdir, rmdir
import glob
from shutil import move

import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, ImageFolder
from torchvision.transforms import v2 as transforms
from torchvision.datasets.utils import download_and_extract_archive

from .DatasetWrapper import DatasetWrapper


def organize_val_folder(target_folder: str):
    """Organize the TinyImageNet validation folder into a structure compatible with ImageFolder loader."""

    # Check if the reorganization has already been done by looking for a flag file
    flag_file_path = os.path.join(target_folder, "val_organized")
    if os.path.exists(flag_file_path):
        print("Validation data has already been organized.")
        return

    val_dict = {}
    with open(os.path.join(target_folder, "val_annotations.txt"), "r") as f:
        for line in f.readlines():
            split_line = line.split("\t")
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob(os.path.join(target_folder, "images", "*"))
    for image_path in paths:
        file_name = os.path.basename(image_path)
        folder_name = val_dict[file_name]
        class_folder_path = os.path.join(target_folder, folder_name)

        if not os.path.exists(class_folder_path):
            os.makedirs(os.path.join(class_folder_path, "images"))

        dest_path = os.path.join(class_folder_path, "images", file_name)
        move(image_path, dest_path)

    os.rmdir(os.path.join(target_folder, "images"))

    # Create a flag file to indicate the organization is complete
    open(flag_file_path, "a").close()
    print("Validation data organization complete.")


class TinyImageNet(VisionDataset):
    num_classes = 200
    base_folder = "tiny-imagenet-200"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        target_transform=None,
        download: bool = False,
    ):
        super(TinyImageNet, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.split = split
        self.root = path.expanduser(root)
        self.dataset_folder = path.join(self.root, self.base_folder)

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        if split == "val":
            organize_val_folder(path.join(self.dataset_folder, "val"))

        if split in ["train", "val"]:
            self.dataset = ImageFolder(
                root=path.join(self.dataset_folder, split), transform=transform
            )
        else:
            raise ValueError(
                f"Unsupported split: {split}. Supported splits are: 'train', 'val'."
            )

    def _download(self):
        if self._check_exists():
            return
        download_and_extract_archive(
            self.url,
            download_root=self.root,
            filename=self.base_folder + ".zip",
            remove_finished=True,
        )

    def _check_exists(self):
        return path.exists(path.join(self.dataset_folder, "train")) and path.exists(
            path.join(self.dataset_folder, "val")
        )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class TinyImageNet_(DatasetWrapper):
    num_classes = 200
    mean = (0.48022226, 0.44811364, 0.39767657)
    std = (0.27139243, 0.26367799, 0.2766458)

    basic_transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    augment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(64),
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
        split = "train" if train else "val"
        self.dataset = TinyImageNet(root, split=split, download=True)
        super().__init__(
            labels=[y for _, y in self.dataset],
            base_ratio=base_ratio,
            num_phases=num_phases,
            augment=augment,
            inplace_repeat=inplace_repeat,
            shuffle_seed=shuffle_seed,
        )
