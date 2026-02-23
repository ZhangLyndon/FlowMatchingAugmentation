import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional

class ResNetClassifier(nn.Module):
	"""
	ResNet-50 classifier for clothing item classification with the Fashion MNIST
	dataset.
	"""
	def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
		"""
		Initialize a ResNet-50 classifier.

		Parameters
		----------
		num_classes : int, optional
			Number of classes in the dataset. Defaults to 10.
		dropout_rate: float, optional
			Probability of zeroing a tensor element in the final fully-connected
			layer. Defaults to 0.5.
		"""
		super().__init__()

		# Load ResNet-50, pretrained on ImageNet, with default (best available)
		# weights.
		self.backbone = resnet50(weights = ResNet50_Weights.DEFAULT)

        # Replace the final fully connected classification layer in the pre-
        # trained ResNet backbone. Retrieve the number of input dimensions,
        # add a dropout layer to prevent overfitting, then apply an affine
        # transformation to the number of output classes.
		feature_dim = self.backbone.fc.in_features
		self.backbone.fc = nn.Sequential(nn.Dropout(dropout_rate),
							        	 nn.Linear(feature_dim, num_classes))

		self.num_classes = num_classes

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Pass input images through the backbone to compute class logits.
		"""
		return self.backbone(x)

	def get_features(self, x):
		"""
		Extract flattened features from ResNet backbone before final classifi-
		cation layer.
		"""
		x = self.backbone.conv1(x)
		x = self.backbone.bn1(x)
		x = self.backbone.relu(x)
		x = self.backbone.maxpool(x)
		x = self.backbone.layer1(x)
		x = self.backbone.layer2(x)
		x = self.backbone.layer3(x)
		x = self.backbone.layer4(x)
		x = self.backbone.avgpool(x)
		return torch.flatten(x, 1)

def create_classifier(num_classes: int = 10, dropout_rate: float = 0.5) -> nn.Module:
	"""
	Instantiate a ResNet classifier with the given number of classes and drop-
	out rate.
	"""
	return ResNetClassifier(num_classes = num_classes, dropout_rate = dropout_rate)

if __name__ == "__main__":
	"""
	Validate model initialization, forward pass logic, and feature extraction.
	"""
	# Initialize the model with the default number of classes (10), and report
	# the total number of trainable parameters.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = create_classifier(num_classes = 10).to(device)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

	# Create a batch of (i.e., 2) 3-channel 224 x 224 images, pass them through
	# the model, and report the shape of class logits.
	x = torch.randn(2, 3, 224, 224)
	output = model(x)
	print(f"Output Shape: {output.shape}")

	# Verify shape of extracted features.
	features = model.get_features(x)
	print(f"Feature Shape: {features.shape}")