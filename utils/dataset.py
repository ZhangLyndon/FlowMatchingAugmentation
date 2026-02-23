import os
from typing import Tuple, Optional

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

def get_dataloaders(root_dir: str, batch_size: int = 32, image_size: int = 224, num_workers: int = 2):
    """
    Create data loaders for the Fashion MNIST dataset, to be used with the
    ResNet-50 classifier.
    """
    # Resize, then add a random horizontal flip to the training set, to enable
    # better generalization. Normalization statistics reflect per-channel va-
    # lues from ImageNet.
    train_transform = transforms.Compose([transforms.Resize(image_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                          transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                               std = [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor(),
                                         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                         transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                              std = [0.229, 0.224, 0.225])])

    # Create training and test sets
    train_dataset = FashionMNIST(root_dir, train = True, download = True, transform = train_transform)
    test_dataset = FashionMNIST(root_dir, train = False, download = True, transform = test_transform)

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
    train_loader, test_loader = get_dataloaders(root_dir)

    print(f"Training Set Size: {len(train_loader.dataset)}")
    print(f"Test Set Size: {len(test_loader.dataset)}")

    # Test a batch within the training set
    for images, labels in train_loader:
        print(f"Batch Shape: {images.shape}")
        print(f"Labels Shape: {labels.shape}")
        print(f"Range of Labels: {labels.min()} - {labels.max()}")