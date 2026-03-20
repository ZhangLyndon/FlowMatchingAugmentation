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
- Modest but consistent accuracy gains (0.45% $-$ 0.8%) when flow model-based synthetic data augmentation is applied to 1% of the original training set
- Larger improvements (3.5% $-$ 4.9%) when augmenting 0.5% of the training data
- Most substantial gains (10.9% $-$ 19.4%) when augmenting only 0.1 $-$ 0.2% of the original training set
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
| 0.5%          | 76.48%                | 81.34%                 | 4.86%                    | 74.49%                | 77.97%                 | 3.48%                    |
| 1%            | 80.62%                | 81.08%                 | 0.46%                    | 80.48%                | 81.29%                 | 0.81%                    |
| 10%           | 88.56%                | 87.41%                 | -1.15%                   | 87.95%                | 87.13%                 | -0.82%                   |
| 100%          | 92.63%                | 91.67%                 | -0.96%                   | 92.13%                | 91.83%                 | -0.3%                    |

We observe the largest gains (> 10%) in the extreme low-data regime, specifically when augmenting 0.1% $-$ 0.2% of the original training set (6-12 samples per class). Increasing the data budget to 0.5% (30 samples per class) yields smaller, but still significant, improvements of 3.5% $-$ 4.9%. At 1% (60 samples per class), synthetic data augmentation provides modest but consistent gains (0.45% $-$ 0.8%), with negligible improvements beyond this point.

This result is particularly notable in light of [prior](https://github.com/Srecharan/GenVision) work, which reported a 4.1% improvement in classification accuracy when augmenting the full training split of the [Caltech-UCSD Birds 200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011) (CUB-200-2011) dataset. CUB-200-2011 is a well-known _fine-grained_ visual classification benchmark comprising approximately 30 samples per class across 200 classes, whereas Fashion MNIST is comparatively coarse-grained. The comparable gains observed here (3.5% $-$ 4.9%) at an equivalent data scale suggest that dataset size, rather than task complexity, is the primary factor driving the effectiveness of synthetic data augmentation.

This trend is further exemplified in the extreme low data regime (0.1% $-$ 0.2%, or 6-12 samples per class), where accuracy gains of 10.9% $-$ 19.4% were observed. Consistently, applying synthetic augmentation to 25% of the training set for the CUB-200-2011 dataset (approximately 7.5 samples per class) resulted in a comparable 8.9% increase in classification accuracy. This implies that the additional task complexity associated with fine-grained classification does not lead to a larger boost in accuracy when augmenting a dataset at similar data scale, as might be naively expected.

Varying the guidance scale used to generate synthetic samples ($w = 3$ versus $w = 5$) has only a minor impact on downstream classification gains. When the guidance scale is higher, samples adhere more strictly to class labels at the expense of diversity. In the extreme low-data regime (< 1%), this reduced variance causes a slight decrease in accuracy improvement, as the greater diversity found at $w = 3$ aids generalization by covering a broader portion of the underlying data distribution.

In contrast, when 1% of the training set is used, the limited diversity of the synthetic samples becomes less problematic. In this regime, the stronger reinforcement of class boundaries at $w = 5$ provides a clearer signal for the classifier, resulting in a marginal increase in accuracy improvement.

### Macro-Averaged $\mathsf F_1$ Score

The macro $\mathsf F_1$ score is the unweighted mean of $\mathsf F_1$ scores across all classes. It is useful for identifying systematic underperformance on individual classes, as all classes are treated equally, regardless of size.

| Data Fraction | Baseline<br>$(w = 3)$ | Augmented<br>$(w = 3)$ | Improvement<br>$(w = 3)$ | Baseline<br>$(w = 5)$ | Augmented<br>$(w = 5)$ | Improvement<br>$(w = 5)$ |
| :-----------: | :-------------------: | :--------------------: | :----------------------: | :-------------------: | :--------------------: | :----------------------: |
| 0.1%          | 0.5731                | 0.7854                 | 0.2123                   | 0.5564                | 0.7432                 | 0.1868                   |
| 0.2%          | 0.6754                | 0.7938                 | 0.1184                   | 0.6536                | 0.7556                 | 0.102                    |
| 0.5%          | 0.7673                | 0.8128                 | 0.0455                   | 0.7474                | 0.7826                 | 0.0352                   |
| 1%            | 0.8023                | 0.8122                 | 0.0099                   | 0.8027                | 0.8138                 | 0.0111                   |
| 10%           | 0.885                 | 0.875                  | -0.01                    | 0.8791                | 0.8726                 | -0.0065                  |
| 100%          | 0.9264                | 0.9161                 | -0.0103                  | 0.9216                | 0.9179                 | -0.0037                  |

The macro $\mathsf F_1$ score follows a similar trend as top-1 accuracy, suggesting that dataset size, rather than task complexity, is the primary factor driving benefits from augmentation. In addition, it indicates no consistent underperformance across individual classes.

## Project Structure

## Method

## Conclusion