import os
import time
import json
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from classification import create_classifier
from utils import get_fgvc_dataloaders, AverageMeter, accuracy, save_results

class ClassificationTrainer:
	"""
	Trainer for aircraft model classification with synthetic data augmentation.
	"""
	def __init__(self, args):
		self.args = args
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.model = create_classifier(num_classes = 100).to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = args.step_size, gamma = args.gamma)

		# Cross entropy loss L = -log[exp(z_y) / sum_k exp(z_k)]
		self.criterion = nn.CrossEntropyLoss()

		# Metrics: best top-1 validation accuracy over all epochs; training loss
		# and top-1 training / validation accuracy for each epoch.
		self.best_acc = 0.0
		self.train_losses = []
		self.train_accs = []
		self.val_accs = []

	def train_epoch(self, train_loader, epoch):
	 	"""
	 	Train the classifier for one epoch.
	 	"""
		self.model.train()

		losses = AverageMeter()
		top1_acc = AverageMeter()
		top5_acc = AverageMeter()

		pbar = tqdm(train_loader, desc = f"Epoch {epoch+1}/{self.args.epochs}")

        for batch_idx, (data, target) in enumerate(pbar):
			data, target = data.to(self.device), target.to(self.device)

			# Forward pass: compute model predictions / loss
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.criterion(output, target)

			# Backward pass: compute gradients of the loss, update weights using
			# the Adam optimizer
			loss.backward()
			self.optimizer.step()
			
			# Fraction of samples where rank of target class is 1 or less than
			# or equal to 5.
            acc1, acc5 = accuracy(output, target, topk = (1, 5))

			pbar.set_postfix({"Loss": f"{losses.avg:.4f}",
							  "Top-1": f"{top1_acc.avg:.2f}%",
							  "Top-5": f"{top5_acc.avg:.2f}%"})

		return losses.avg, top1_acc.avg, top5_acc.avg

    def validate(self, val_loader, epoch):
        """
        Validate model performance.
        """
        self.model.eval()
        
        losses = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

		with torch.no_grad():
			for data, target in tqdm(val_loader, desc = "Validation"):
				data, target = data.to(self.device), target.to(self.device)

				# Forward pass: compute model prediction and cross-entropy loss
				output = self.model(data)
				loss = self.criterion(output, target)

				# Fraction of samples where rank of target class is 1 or less
				# than or equal to 5.
				acc1, acc5 = accuracy(output, target, topk = (1, 5))

				# Update average loss, and top-1 (or 5) categorical accuracy.
				losses.update(loss.item(), data.size(0))
				top1_acc.update(acc1[0], data.size(0))
				top5_acc.update(acc5[0], data.size(0))

        return losses.avg, top1_acc.avg, top5_acc.avg

	def save_checkpoint(self, epoch, val_acc, is_best = False):
        """
        Save the ResNet classifier state as a checkpoint.
        """
        state = {"epoch": epoch,
		         "model_state_dict": self.model.state_dict(),
		         "optimizer_state_dict": self.optimizer.state_dict(),
		         "scheduler_state_dict": self.scheduler.state_dict(),
		         "best_acc": self.best_acc,
		         "val_acc": val_acc}

		checkpoint_path = os.path.join(self.args.checkpoint_dir, f"resnet_epoch_{epoch}.pt")
		torch.save(state, checkpoint_path)

		if is_best:
			best_path = os.path.join(self.args.checkpoint_dir, "resnet_best_val_top1.pt")
			torch.save(state, best_path)

    def train(self, train_loader, val_loader):
        """
	    Train the model.
        """
        print(f"Number of epochs: {self.args.epochs}")
        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of validation samples: {len(val_loader.dataset)}")

	    start_time = time.time()
        for epoch in range(self.args.epochs):
        	# Train the classifier for one epoch, returning the average loss
        	# and top-1 / 5 categorical accuracy.
			train_loss, train_top1, train_top5 = self.train_epoch(train_loader, epoch)

            # Validate model performance to assess generalization. We are using
            # the test split here since we're not doing hyperparameter tuning or
            # model selection.
            val_loss, val_top1, val_top5 = self.validate(val_loader, epoch)

			# Step the scheduler. Starting at an alpha (learning rate) of 0.001,
			# we scale it down by 0.1 every 15 epochs to transition from broad
			# exploration to fine-tuned exploitation.
			self.scheduler.step()

			# Add training loss and top-1 training / validation accuracy for this
			# epoch to their respective lists.
			self.train_losses.append(train_loss)
			self.train_accs.append(train_top1)
			self.val_accs.append(val_top1)

			# Best model is defined by the top-1 accuracy (proportion of samples
			# where the top-scoring class matches the ground truth) on the vali-
			# dation set.
			is_best = val_top1 > self.best_acc
			if is_best:
				self.best_acc = val_top1

			# Save model state as a checkpoint every N epochs, or if validation
			# top-1 accuracy reaches a new peak.
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
	            self.save_checkpoint(epoch, val_top1, is_best)

			# Print summary statistics for epoch
			print(f"[Epoch {epoch + 1}/{self.args.epochs}]")
			print("Training Set")
			print(f"Loss: {train_loss:.4f}, Top-1 Accuracy: {train_top1:.2f}%, Top-5 Accuracy: {train_top5:.2f}%")
			print("Validation Set")
			print(f"Loss: {val_loss:.4f}, Top-1 Accuracy: {val_top1:.2f}%, Top-5 Accuracy: {val_top5:.2f}%")
			print(f"Best Top-1 Validation Accuracy (Up Until Now): {self.best_acc:.2f}%")
			print("-" * 60)

		total_time = time.time() - start_time
		print(f"Training Time: {total_time:.2f} seconds")
		print(f"Best Top-1 Validation Accuracy (All Epochs): {self.best_acc:.2f}%")

		# Save best top-1 validation accuracy, final top-1 training accuracy,
		# training losses / top-1 training / validation accuracies over all
		# epochs, as well as training time and number of epochs completed.
        results = {"epochs_completed": len(self.train_losses),
			       "training_time_seconds": total_time,
			       "best_val_accuracy": self.best_acc,
			       "final_train_accuracy": train_top1,
			       "train_losses": self.train_losses,
			       "train_accuracies": self.train_accs,
			       "val_accuracies": self.val_accs}

		# Save training results as a JSON file.
		# Default path: ./results/classification/training_results.json.
		results_path = os.path.join(self.args.classification_dir, "training_results.json")
		save_results(results, results_path)
        
        return results

def parse_args():
	parser = argparse.ArgumentParser(description = "Train ResNet-50 Classifier on FGVC Aircraft", add_help = False)
	
	# Data
	parser.add_argument("--data_root", type = str, default = "/content/data", help = "Path to FGVC Aircraft dataset")
	parser.add_argument("--batch_size", type = int, default = 32, help = "Training batch size")
	parser.add_argument("--num_workers", type = int, default = 2, help = "Number of data loading workers")

	# Training
	parser.add_argument("--epochs", type = int, default = 50, help = "Number of classifier training epochs")
	parser.add_argument("--lr", type = float, default = 0.001, help = "Learning rate")
    parser.add_argument("--weight_decay", type = float, default = 1e-4, help = "Weight decay")
    parser.add_argument("--step_size", type = int, default = 15, help = "Scheduler step size")
    parser.add_argument("--gamma", type = float, default = 0.1, help = "Scheduler gamma")

    # Output
    parser.add_argument("--classification_dir", type = str, default = "/content/results/classification",
				    	help = "Directory for classification results")
    parser.add_argument("--checkpoint_dir", type = str, default = "/content/checkpoints",
				    	help = "Directory for model checkpoints")
    parser.add_argument("--save_interval", type = int, default = 5, help = "Number of epochs between checkpoint saves")

	# General
	parser.add_argument("--seed", type = int, default = 42, help = "Random seed")

	return parser.parse_args()

def main():
	args = parse_args()
	
	# Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories for classification results and model checkpoints
	os.makedirs(args.classification_dir, exist_ok = True)
    os.makedirs(args.checkpoint_dir, exist_ok = True)

	train_loader, test_loader = get_fgvc_dataloaders(root_dir = args.data_root,
													 batch_size = args.batch_size,
													 num_workers = args.num_workers)

	# Initialize the ResNet-50 trainer on FGVC Aircraft. Monitor performance
	# using the test split.
	trainer = ClassificationTrainer(args)
	results = trainer.train(train_loader, test_loader)

if __name__ == "__main__":
	main()