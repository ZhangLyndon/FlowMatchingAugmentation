import os
from typing import Tuple, Optional

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

def get_dataloaders(root_dir: str, batch_size: int = 16, image_size: int = 32, num_workers: int = 0):
    """
    Create data loaders for the Fashion MNIST dataset, to be used with the
    ResNet-18 classifier.
    """
    # Resize, then add a random horizontal flip to the training set, to enable
    # better generalization. Normalization statistics Fashion MNIST per-channel
    # values.
    train_transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.28604],
                                                               std = [0.35302])])

    test_transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean = [0.28604],
                                                              std = [0.35302])])

    # Create training and test sets
    train_dataset = FashionMNIST(root_dir, train = True, download = True, transform = train_transform)
    test_dataset = FashionMNIST(root_dir, train = False, download = True, transform = test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size = batch_size,
                              shuffle = True,
                              num_workers = num_workers)
    
    test_loader = DataLoader(test_dataset,
                             batch_size = batch_size,
                             shuffle = False,
                             num_workers = num_workers)

    return train_loader, test_loader

if __name__ == "__main__":
    # Test the data loaders
    root_dir = "./data"
    train_loader, test_loader = get_dataloaders(root_dir)

    print(f"Training Set Size: {len(train_loader.dataset)}")
    print(f"Test Set Size: {len(test_loader.dataset)}")

    # Test a batch within the training set
    for images, labels in train_loader:
        print(f"Batch Shape: {images.shape}")
        print(f"Labels Shape: {labels.shape}")
        print(f"Range of Labels: {labels.min()} - {labels.max()}")