# -*- coding: utf-8 -*-
import os
import shutil
import zipfile
import tarfile
from os import path, makedirs

import requests
import torch
from tqdm import tqdm
from torchvision.datasets import Caltech256, Caltech101, ImageFolder
from torchvision.transforms import v2 as transforms
from typing import Tuple

from .DatasetWrapper import DatasetWrapper


def download_and_extract(url: str, download_path: str):
    """Download and extract the dataset from a given URL, displaying download progress. It will also check and adjust the directory structure if necessary."""
    makedirs(download_path, exist_ok=True)
    file_name = url.split("/")[-1]
    file_path = path.join(download_path, file_name)

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    # Check file extension and extract
    if file_path.endswith(".tar.gz") or file_path.endswith(".tar"):
        print(f"Extracting {file_name}...")
        with tarfile.open(file_path, "r:*") as tar:
            tar.extractall(path=download_path)
    elif file_path.endswith(".zip"):
        print(f"Extracting {file_name}...")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(download_path)
            # Check and extract .tar.gz files inside the zip directory
            inside_dir = file_path.replace(".zip", "")
            for item in os.listdir(inside_dir):  # Iterate through the folder
                if item.endswith(".tar.gz"):
                    print(f"Extracting {item} inside the zip...")
                    tar_path = path.join(inside_dir, item)
                    with tarfile.open(tar_path, "r:*") as tar:
                        tar.extractall(path=inside_dir)
                    os.remove(tar_path)  # Delete the .tar.gz file after extraction

    print(f"Downloaded and extracted dataset to {download_path}")


class Caltech256_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 257
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
        dataset_path = os.path.join(root, "256_ObjectCategories")
        dataset_downloaded = False

        if os.path.exists(dataset_path) and os.listdir(dataset_path):
            print(
                f"Dataset already exists at {dataset_path}. Using the existing dataset."
            )
            dataset_downloaded = True
            self.dataset = ImageFolder(dataset_path)

        if not dataset_downloaded:
            try:
                self.dataset = Caltech256(root, download=True)
                dataset_downloaded = True
            except Exception as e:
                print(f"Downloading from torchvision failed: {e}.")

            if not dataset_downloaded:
                print("Attempting to download from a backup URL...")
                backup_url = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar"
                download_and_extract(backup_url, root)
                self.dataset = ImageFolder(dataset_path)

        super().__init__(
            [item[1] for item in self.dataset.samples],
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )


class Caltech101_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 102
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
        # Paths for backup URL download and extraction
        extracted_inner_path_backup = os.path.join(
            root, "caltech-101", "101_ObjectCategories"
        )
        # Path for torchvision download
        extracted_inner_path_torchvision = os.path.join(root, "101_ObjectCategories")

        dataset_downloaded = False

        # Check backup URL download path first
        if os.path.exists(extracted_inner_path_backup) and os.listdir(
            extracted_inner_path_backup
        ):
            print(
                f"Dataset already exists at {extracted_inner_path_backup}. Using the existing dataset."
            )
            dataset_downloaded = True
            self.dataset = ImageFolder(extracted_inner_path_backup)
        # Then check torchvision download path
        elif os.path.exists(extracted_inner_path_torchvision) and os.listdir(
            extracted_inner_path_torchvision
        ):
            print(
                f"Dataset already exists at {extracted_inner_path_torchvision}. Using the existing dataset."
            )
            dataset_downloaded = True
            self.dataset = ImageFolder(extracted_inner_path_torchvision)

        if not dataset_downloaded:
            try:
                self.dataset = Caltech101(root, download=True)
                dataset_downloaded = True
            except Exception as e:
                print(f"Downloading from torchvision failed: {e}.")

            if not dataset_downloaded:
                print("Attempting to download from a backup URL...")
                backup_url = (
                    "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
                )
                download_and_extract(backup_url, root)
                if os.path.exists(extracted_inner_path_backup) and os.listdir(
                    extracted_inner_path_backup
                ):
                    self.dataset = ImageFolder(extracted_inner_path_backup)
                elif os.path.exists(extracted_inner_path_torchvision) and os.listdir(
                    extracted_inner_path_torchvision
                ):
                    self.dataset = ImageFolder(extracted_inner_path_torchvision)

        super().__init__(
            [item[1] for item in self.dataset.samples],
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )
