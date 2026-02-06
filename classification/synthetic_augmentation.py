import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

# ConcatDataset allows merging multiple datasets into one
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# DataLoaders for FGVC Aircraft, training class for ResNet classifier, and
# U-Net / classifier-free guided vector field for image generation
from utils import get_fgvc_dataloaders
from classification import ClassificationTrainer
from flow.models import ConditionalVectorField, CFGVectorFieldODE, UNet

def parse_args():
	parser = argparse.ArgumentParser(description = "Evaluation of Synthetic Data Augmentation", add_help = False)

	# Data
	parser.add_argument("--data_root", type = str, default = "/content/data", help = "Path to FGVC Aircraft dataset")
	parser.add_argument("--batch_size", type = int, default = 32, help = "Training batch size")
	parser.add_argument("--num_workers", type = int, default = 2, help = "Number of data loading workers")

	# Training
	parser.add_argument("--epochs", type = int, default = 20, help = "Number of training epochs")
	parser.add_argument("--data_fraction", type = float, default = 1.0, help = "Fraction of training data to use")

	# Output
	parser.add_argument("--results_dir", type = str, default = "/content/results/augmentation",
						help = "Directory for augmentation results")

	# Experiments
	parser.add_argument("--run_low_data", action = "store_true", help = "Run low-data experiments")

	return parser.parse_args()

def main():
	args = parse_args()

if __name__ == "__main__":
	main()