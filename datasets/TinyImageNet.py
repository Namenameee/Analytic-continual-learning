# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
from os import path
import os
import glob
from shutil import move
import requests
from zipfile import ZipFile
from .ImageNet import ImageNet_


def download_and_extract(url: str, dest_folder: str):
    """Download and extract the TinyImageNet zip file."""
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = path.join(dest_folder, "tiny-imagenet-200.zip")

    if not path.exists(zip_path):
        print(f"Downloading {url} to {zip_path}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192  # 8 Kibibytes
        progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                file.write(chunk)
        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR: Something went wrong during the download.")
        else:
            print("Download complete.")

    extract_path = path.join(dest_folder, "tiny-imagenet-200")
    if not path.exists(extract_path):
        print(f"Extracting {zip_path} to {extract_path}")
        with ZipFile(zip_path, "r") as zip_ref:
            for file in tqdm(
                iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), unit="file"
            ):
                zip_ref.extract(member=file, path=dest_folder)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")


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


class TinyImageNet_(ImageNet_):
    num_classes = 200
    mean = (0.48022226, 0.44811364, 0.39767657)
    std = (0.27139243, 0.26367799, 0.2766458)

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
        base_folder = "tiny-imagenet-200"
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        updated_root = path.join(root, base_folder)

        download_and_extract(url, root)

        if train:
            dataset_path = path.join(updated_root, "train")
        else:
            dataset_path = path.join(updated_root, "val")
            organize_val_folder(dataset_path)

        self.dataset = ImageFolder(root=dataset_path)

        super(ImageNet_, self).__init__(
            self.dataset.targets,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )
