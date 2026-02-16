import os
import argparse
from typing import Tuple, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

from flow.models import (CFGVectorFieldODE,
						 CFGTrainer,
						 FGVCSampler,
						 GaussianConditionalProbabilityPath,
						 IsotropicGaussian,
						 UNet)
from flow.utils import LinearAlpha, LinearBeta

from flow import EulerSimulator
from utils import get_fgvc_dataloaders

class FlowMatchingPipeline:
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Create output image directory
		sample_dir = "/content/images"
		os.makedirs(sample_dir, exist_ok = True)

		# Initialize DataLoaders for the training and test sets
		root_dir = "/content/data"
		self.train_loader, self.test_loader = get_fgvc_dataloaders(root_dir)

		# Initialize Gaussian conditional probability path to guide samples from
		# an isotropic multivariate Gaussian to the data distribution.
		self.sampler = FGVCSampler(self.train_loader).to(self.device)
		self.path = GaussianConditionalProbabilityPath(p_data = self.sampler,
													   p_simple_shape = [3, 224, 224],
													   alpha = LinearAlpha(),
													   beta = LinearBeta()).to(self.device)

		# Initialize the U-Net and CFG (classifier-free guidance) trainer
		self.unet = UNet(channels = [32, 64, 128, 256],
						 num_residual_layers = 2,
						 t_embed_dim = 128,
						 y_embed_dim = 128)

		self.trainer = CFGTrainer(path = self.path, model = self.unet, eta = 0.1, null_label = 100)
	
	def generate_samples(self,
						 samples_per_class: int = 67,
						 guidance_scales: Tuple[float, ...] = (3.0, 5.0, 7.0),
						 num_timesteps: int = 100) -> Dict[float, Tuple[torch.Tensor, torch.Tensor]]:

		# Train the U-Net for the specified number of epochs and learning rate.
		self.trainer.train(num_epochs = 300, device = self.device, lr = 1e-4, batch_size = 32)
		
		# Create a samples dictionary, keyed by the guidance scale w, to store
		# generated samples.
		samples = {}
		num_classes = 100

		for w in guidance_scales:
			# Create a classifier-free guided vector field with the specified
			# guidance scale, then simulate as an ODE.
			ode = CFGVectorFieldODE(self.unet, guidance_scale = w)
			simulator = EulerSimulator(ode)

			# Create copies of each class label (0, ..., 99). Exclude the null
			# label from generated samples.
			y = torch.arange(num_classes, dtype = torch.int64).repeat_interleave(samples_per_class).to(self.device)
			num_samples = y.shape[0]

			# Draw samples from the isotropic Gaussian as p_noise.
			x0, _ = self.path.p_simple.sample(num_samples)
			ts = torch.linspace(0, 1, num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(self.device)

			# Simulate for the given number of time steps and guidance scale,
			# returning the final state x1 and corresponding class labels y.
			# y is of shape (num_samples, ), while x1 has shape (num_samples,
			# 3, 224, 224). To filter for a class, create a mask using mask =
			# (y == class), then apply it using x1_class = x1[mask].
			x1 = simulator.simulate(x0, ts, y = y)
			samples[w] = (x1, y)

		return samples

	def postprocessing(self, samples: Dict[float, torch.Tensor]):
		# Transform to remove ImageNet normalization
		denorm = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
									  std = [1/0.229, 1/0.224, 1/0.225])

		# Remove ImageNet normalization and clamp values to [0, 1] range.
		for w, (x1, y) in samples.items():
			x1 = denorm(x1).clamp(0, 1)
			samples[w] = (x1, y)
			num_samples = y.shape[0]

			for index in range(num_samples):
				image = x1[index]
				class_label = y[index].item()
				# Organize images based on the guidance scale and class label
				directory = f"/content/images/w-{self.guidance_scale}/class-{class_label}"
				os.makedirs(directory, exist_ok = True)

				file_name = f"image-{index}_w-{w}_class-{class_label:02d}.png"
				file_path = os.path.join(directory, file_name)
				save_image(image, fp = file_path)

	def save_checkpoint(self, checkpoint_dir):
		"""
		Save the U-Net state as a checkpoint.
		"""
		os.makedirs(checkpoint_dir, exist_ok = True)
		checkpoint_path = os.path.join(checkpoint_dir, "unet.pt")

		state = {"model_state_dict": self.unet.state_dict()}
		torch.save(state, checkpoint_path)

def main():
	parser = argparse.ArgumentParser(description = "Flow Matching Pipeline", add_help = False)
	parser.add_argument("--num_samples", type = int, default = 67, help = "Number of samples to generate")
	parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
	parser.add_argument("--checkpoint_dir", type = str, default = "/content/checkpoints",
						help = "Directory for model checkpoints")

	args = parser.parse_args()

	# Set random seed
	torch.manual_seed(args.seed)

	# Initialize training and inference
	flow = FlowMatchingPipeline()

	# Generate samples using classifier-free guidance conditional flow matching,
	# then save the generated samples. Persist the U-Net weights as a checkpoint.
	samples = flow.generate_samples(samples_per_class = args.num_samples)
	flow.postprocessing(samples)
	flow.save_checkpoint(args.checkpoint_dir)

if __name__ == "__main__":
	main()