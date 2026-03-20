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

This project implements and validates synthetic data augmentation for computer vision using conditional flow matching (CFM) with classifier-free guidance (CFG), a state-of-the-art generative modeling technique. Our results indicate that incorporating synthetically generated samples into training improves classification accuracy and $\mathsf F_1$ score for coarse-grained fashion item classification compared with models trained solely on real data, with the largest gains observed in the extreme low-data regime, i.e., < 1% of the full dataset.

**Key Results**
- Modest but consistent accuracy improvements (0.45% $-$ 0.8%) with flow model-based synthetic data augmentation at 1% of the training set
- Pronounced gains (10.9% $-$ 19.4%) in the extreme low-data regime, i.e., < 1% of the training set
- 60,000 synthetic images generated across 10 fashion item categories
- Evaluation using ResNet-18 (pre-trained on ImageNet), fine-tuned on fractions of the training set, with and without synthetic augmentation

The implementation trains a conditional flow matching model on the training split of the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, using classifier-free guidance to generate class-conditioned synthetic samples. A ResNet-18 classifier, pre-trained on ImageNet, is then fine-tuned on varying fractions of the training data, with and without synthetic augmentation, to evaluate the impact of data augmentation. Classification performance is assessed using top-1 accuracy and the macro-averaged $\mathsf F_1$ score.

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

### Top-1 Accuracy

Top-1 accuracy represents the proportion of samples for which the model's highest-scoring predicted class $-$ i.e., the class with the largest logit or predicted probability $-$ matches the ground truth label. It is equivalent to standard classification accuracy, when evaluations are based on the highest-scoring class for each input.

| Data Fraction | Baseline<br>$(w = 3)$ | Augmented<br>$(w = 3)$ | Improvement<br>$(w = 3)$ | Baseline<br>$(w = 5)$ | Augmented<br>$(w = 5)$ | Improvement<br>$(w = 5)$ |
| :-----------: | :-------------------: | :--------------------: | :----------------------: | :-------------------: | :--------------------: | :----------------------: |
| 0.1%          | 59.07%                | 78.43%                 | 19.36%                   | 57.88%                | 74.51%                 | 16.63%                   |
| 0.2%          | 66.61%                | 79.56%                 | 12.95%                   | 64.65%                | 75.59%                 | 10.94%                   |
| 1%            | 80.62%                | 81.08%                 | 0.46%                    | 80.48%                | 81.29%                 | 0.81%                    |
| 10%           | 88.56%                | 87.41%                 | -1.15%                   | 87.95%                | 87.13%                 | -0.82%                   |
| 100%          | 92.63%                | 91.67%                 | -0.96%                   | 92.13%                | 91.83%                 | -0.3%                    |

We observe the largest gains (> 10%) in the extreme low-data (< 1%) regime, in particular at 0.1% $-$ 0.2% of the training set, where performance improves substantially with only 6-12 samples per class. Fine-tuning on 1% of the training set yields modest but consistent improvements (0.45% $-$ 0.8%), corresponding to around 60 samples per class, with negligible gains beyond this point.

## Project Structure

## Method

## Conclusion