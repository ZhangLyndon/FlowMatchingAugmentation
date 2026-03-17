<div align = "center">

# Flow Matching for Synthetic Image Augmentation

Flow Matching Synthetic Image Generation for Data Augmentation

[**Overview**](#overview)
| [**Usage**](#usage)
| [**Results**](#results)
| [**Project Structure**](#project-structure)
| [**Method**](#method)
| [**Conclusion**](#conclusion)

</div>

## Overview

This project implements and validates synthetic data augmentation for computer vision using conditional flow matching (CFM) with classifier-free guidance (CFG), a state-of-the-art generative modeling technique. Our results indicate that incorporating synthetically generated samples into training improves classification accuracy and $\mathsf F_1$ score for coarse-grained fashion item classification compared with models trained solely on real data, with the largest gains observed in the extreme low-data regime, i.e., $<1\\%$ of the full dataset.

**Key Results**
- Modest but consistent accuracy improvements $\left(0.45\\%-0.8\\%\right)$ with flow model-based synthetic data augmentation at $1\\%$ of the training set
- Pronounced gains $\left(10.9\\%-19.4\\%\right)$ in the extreme low-data regime, i.e., $<1\\%$ of the training set
- 60,000 synthetic images generated across 10 fashion item categories
- Evaluation using ResNet-18 (pre-trained on ImageNet), fine-tuned on fractions of the training set, with and without synthetic augmentation

The implementation trains a conditional flow matching model on the training split of the Fashion MNIST dataset, using classifier-free guidance to generate class-conditioned synthetic samples. A ResNet-18 classifier, pre-trained on ImageNet, is then fine-tuned on varying fractions of the training data, with and without synthetic augmentation, to evaluate the impact of data augmentation. Classification performance is assessed using top-1 accuracy and the macro-averaged $\mathsf F_1$ score.

## Usage

1. Clone the repository.
```bash
git clone https://github.com/ZhangLyndon/FlowMatchingAugmentation .
```

2. Install dependencies.
```bash
pip install -r requirements.txt
```

3. Generate synthetic samples.
```bash
python flow/pipeline.py --num_samples 6000 \
						--generation_batch 6000 \
						--checkpoint_dir ./checkpoints
```

4. Fine-tune the pre-trained ResNet on the full training split of the Fashion MNIST dataset, using a held-out validation set to monitor cross-entropy loss and determine the optimal number of epochs to train for before overfitting.
```bash
python classification/train_classifier.py --data_root ./data \
										  --batch_size 16 \
										  --num_workers 0 \
										  --epochs 30 \
										  --lr 0.001 \
										  --weight_decay 1e-4 \
										  --step_size 15 \
										  --gamma 0.1 \
										  --classification_dir ./results/classification \
										  --checkpoint_dir ./checkpoints \
										  --save_interval 10
```

5. Evaluate the performance of the trained baseline classifier on the Fashion MNIST test set using top-1 accuracy and the macro $\mathsf F_1$ score. Subsequently, evaluate the effect of synthetic image augmentation on classification performance by fine-tuning on fractions of the training set with and without synthetic samples.
```bash
python classification/synthetic_augmentation.py --data_root ./data \
												--batch_size 16 \
												--num_workers 0 \
												--epochs 25 \
												--lr 0.001 \
												--weight_decay 1e-4 \
												--step_size 15 \
												--gamma 0.1 \
												--synthetic_data_dir ./images \
												--real_ratio 0.001 \
												--classification_dir ./results/classification \
												--augmentation_dir ./results/augmentation \
												--checkpoint_dir ./checkpoints \
												--save_interval 20
```

## Results

## Project Structure

## Method

## Conclusion