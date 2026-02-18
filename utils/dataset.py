import os
from typing import Tuple, Optional

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FGVCAircraft

from PIL import Image

def remove_banner(image: Image.Image) -> Image.Image:
    """
    Removes the bottom 20 pixels from an image represented by a PIL Image.
    Assumes image is a tensor of shape (channels, height, width).
    """
    width, height = image.size
    return image.crop((0, 0, width, height - 20))

def get_fgvc_dataloaders(root_dir: str, batch_size: int = 32, image_size: int = 224, num_workers: int = 2):
    """
    Create data loaders for the FGVC Aircraft dataset.
    """
    # Remove the banner at the bottom displaying copyright information. Resize,
    # then crop the image at a random location to prevent memorization. Add a
    # random horizontal flip to the training set, to enable better generaliza-
    # tion. Normalization statistics reflect per-channel values from ImageNet.
    train_transform = transforms.Compose([transforms.Lambda(remove_banner),
                                          transforms.Resize(image_size + 32),
                                          transforms.RandomCrop(image_size), 
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                               std = [0.229, 0.224, 0.225])])
    
    # For the test set, cropping the image at the center helps ensure determi-
    # nistic evaluation.
    test_transform = transforms.Compose([transforms.Lambda(remove_banner),
                                         transforms.Resize(image_size + 32),
                                         transforms.CenterCrop(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                              std = [0.229, 0.224, 0.225])])

    # Create training and test sets
    train_dataset = FGVCAircraft(root_dir, split = "trainval", download = True, transform = train_transform)
    test_dataset = FGVCAircraft(root_dir, split = "test", download = True, transform = test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size = batch_size,
                              shuffle = True,
                              num_workers = num_workers,
                              pin_memory = True,
                              persistent_workers = True)
    
    test_loader = DataLoader(test_dataset,
                             batch_size = batch_size,
                             shuffle = False,
                             num_workers = num_workers,
                             pin_memory = True,
                             persistent_workers = True)

    return train_loader, test_loader

if __name__ == "__main__":
    # Test the data loaders
    root_dir = "/content/data"
    train_loader, test_loader = get_fgvc_dataloaders(root_dir)

    print(f"Training Set Size: {len(train_loader.dataset)}")
    print(f"Test Set Size: {len(test_loader.dataset)}")

    # Test a batch within the training set
    for images, labels in train_loader:
        print(f"Batch Shape: {images.shape}")
        print(f"Labels Shape: {labels.shape}")
        print(f"Range of Labels: {labels.min()} - {labels.max()}")