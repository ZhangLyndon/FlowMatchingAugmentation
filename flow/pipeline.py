import os
import gc
import math
import argparse
from typing import Tuple, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

from flow.models import (CFGVectorFieldODE,
						 CFGTrainer,
						 Sampler,
						 GaussianConditionalProbabilityPath,
						 IsotropicGaussian,
						 UNet)
from flow.utils import LinearAlpha, LinearBeta
from flow import EulerSimulator

class FlowMatchingPipeline:
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Create output image directory
		sample_dir = "./images"
		os.makedirs(sample_dir, exist_ok = True)

		# Initialize Gaussian conditional probability path to guide samples from
		# an isotropic multivariate Gaussian to the data distribution.
		self.sampler = Sampler().to(self.device)
		self.path = GaussianConditionalProbabilityPath(p_data = self.sampler,
													   p_simple_shape = [1, 32, 32],
													   alpha = LinearAlpha(),
													   beta = LinearBeta()).to(self.device)

		# Initialize the U-Net and CFG (classifier-free guidance) trainer. Move
		# the model to the GPU if available.
		self.unet = UNet(channels = [32, 64, 128],
						 num_residual_layers = 2,
						 t_embed_dim = 40,
						 y_embed_dim = 40).to(self.device)
		if torch.cuda.device_count() > 1:
			self.unet = nn.DataParallel(self.unet)

		self.trainer = CFGTrainer(path = self.path, model = self.unet, eta = 0.1, null_label = 10)
	
	def generate_samples(self,
						 samples_per_class: int = 6000,
						 generation_batch: int = 6000,
						 guidance_scales: Tuple[float, ...] = (3.0, 5.0),
						 num_timesteps: int = 100):

		# Train the U-Net for the specified number of epochs and learning rate.
		self.trainer.train(num_epochs = 6000, device = self.device, lr = 1e-3, batch_size = 250)

		# Clear memory post-training.
		torch.cuda.empty_cache()
		gc.collect()
		
		num_classes = 10
		for w in guidance_scales:
			# Create a classifier-free guided vector field with the specified
			# guidance scale, then simulate as an ODE.
			ode = CFGVectorFieldODE(self.unet, guidance_scale = w)
			simulator = EulerSimulator(ode)

			# Iterate over each class label (0, ..., 9), excluding the null label.
			# Generate samples for each class one batch at a time to avoid out of
			# memory errors.
			for class_label in range(num_classes):
				for batch_id in range(math.ceil(samples_per_class / generation_batch)):
					# Calculate start and end indices for the current batch
					start_index = batch_id * generation_batch
					end_index = min((batch_id + 1) * generation_batch, samples_per_class)
					num_samples = end_index - start_index

					y = torch.full((num_samples, ), class_label, dtype = torch.int64).to(self.device)
					# Draw samples from the isotropic Gaussian as p_noise.
					x0, _ = self.path.p_simple.sample(num_samples)
					ts = torch.linspace(0, 1, num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(self.device)

					# Simulate for the given number of time steps and guidance scale.
					x1 = simulator.simulate(x0, ts, y = y)

					# Postprocess and save the batch of samples.
					for sample_id in range(start_index, end_index):
						self.postprocessing(x1[sample_id - start_index], w, class_label, sample_id)

					# Delete tensors and clear memory.
					del x0, x1, ts, y
					torch.cuda.empty_cache()
					gc.collect()

	def postprocessing(self, sample: torch.Tensor, guidance_scale: float, class_label: int, sample_id: int):
		# Remove normalization and clamp values to [0, 1] range.
		denorm = transforms.Normalize(mean = [-0.5/0.5],
									  std = [1/0.5])
		sample = denorm(sample).clamp(0, 1)

		# Create directory to store the generated sample. Organize based on the
		# guidance scale and class label.
		directory = f"./images/w-{guidance_scale}/class-{class_label}"
		os.makedirs(directory, exist_ok = True)

		# Save the image to disk.
		file_name = f"image-{sample_id}_w-{guidance_scale}_class-{class_label}.png"
		file_path = os.path.join(directory, file_name)
		save_image(sample, fp = file_path)

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
	parser.add_argument("--num_samples", type = int, default = 6000, help = "Number of samples to generate per class")
	parser.add_argument("--generation_batch", type = int, default = 6000, help = "Number of samples to generate at a time")
	parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
	parser.add_argument("--checkpoint_dir", type = str, default = "./checkpoints",
						help = "Directory for model checkpoints")

	args = parser.parse_args()

	# Set random seed
	torch.manual_seed(args.seed)

	# Initialize training and inference
	flow = FlowMatchingPipeline()

	# Generate samples using classifier-free guidance conditional flow matching,
	# then save the generated samples. Persist the U-Net weights as a checkpoint.
	flow.generate_samples(samples_per_class = args.num_samples, generation_batch = args.generation_batch)
	flow.save_checkpoint(args.checkpoint_dir)

if __name__ == "__main__":
	main()