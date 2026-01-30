import os
import json
import torch
import numpy as np

class AverageMeter:
    """
    Meter to track values (and their averages) during training, validation, and
    inference.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk = (1, )):
    """
    Computes the top-k categorical accuracy for single-label classification,
    i.e., the fraction of samples where the rank of the target class is less
    than or equal to k.

    Parameters
    ----------
    output : torch.Tensor, shape (batch_size, num_classes)
        Raw model logits / scores.
    target : torch.Tensor, shape (batch_size, )
        Ground truth class indices.
    topk : tuple[int, ...]
        Tuple of k values over which to subset predictions, to validate against
        ground truth values.

    Returns
    -------
    res : list of torch.Tensor
        Top-k accuracies (shape [1]), as percentages.
    """
    # Disable gradient computation during validation and inference.
    with torch.no_grad():
        # Retrieve the batch size and maximum k value.
        maxk = max(topk)
        batch_size = target.size(0)
        
        # Retrieve the indices corresponding to the top-k scores or logits,
        # then transpose. Yields a tensor of shape (maxk, batch_size).
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # Reshape the ground truth values to (maxk, batch_size), then compare
        # element-wise with the predictions.
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            # For each k, count the number of samples where the target appears
            # among the top k predictions. Sum over the number of correct pre-
            # dictions. At most 1 of the top k predictions for each sample in
            # the batch will be correct, so the accuracy percentage can be com-
            # puted by dividing by the batch size.
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def compute_average_metrics(metrics_list):
    """
    Compute the average of a set of metrics across batches.

    Parameters
    ----------
    metrics_list : list[dict]
        List of per-batch dictionaries, each containing metrics.

    Returns
    -------
    avg_metrics : dict
        Dictionary with averaged metrics.
    """
    # Return an empty dictionary if the list is empty.
    if not metrics_list:
        return {}

    avg_metrics = {}
    for key in metrics_list[0].keys():
        # Average each metric across all dicts sharing that key. Missing keys
        # are ignored.
        values = [m[key] for m in metrics_list if key in m]
        if values:
            avg_metrics[key] = sum(values) / len(values)

    return avg_metrics

def save_results(results, filepath):
    """
    Save results to JSON file.
    """
    # Extract the directory from the file path, raising no errors if it already
    # exists.
    os.makedirs(os.path.dirname(filepath), exist_ok = True)
    # Serialize the results to a JSON file, with a 2-space indent.
    with open(filepath, "w") as f:
        json.dump(results, f, indent = 2)

def load_results(filepath):
    """
    Load results from a JSON file.
    """
    with open(filepath, "r") as f:
        return json.load(f)

def calculate_classification_metrics(predictions, targets, num_classes = 100):
    """
    Calculate classification metrics, such as top 1 / 5 and per-class accuracy.
    
    Parameters
    ----------
    predictions : torch.Tensor, shape (batch_size, num_classes)
        Model predictions, e.g., logits or scores.
    targets : torch.Tensor, shape (batch_size, )
        Ground truth class indices.
    num_classes : int
        Total number of classes.
    
    Returns
    -------
    metrics : dict
        Dictionary of accuracies and sample count.
    """
    with torch.no_grad():
        # Convert PyTorch tensors to NumPy arrays, if necessary.
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        # Predicted class corresponds to the column index for the maximum logit
        # or score.
        pred_classes = np.argmax(predictions, axis = 1)

        # Overall accuracy: percentage of samples for which the class with the
        # maximum logit or score corresponds to the ground truth.
        accuracy_score = np.mean(pred_classes == targets) * 100

        # Top 5 accuracy: return column indices of the 5 highest scoring classes
        # for each sample. Check whether the ground truth corresponds to one of
        # these, then compute the percentage accuracy over the batch.
        top5_preds = np.argsort(predictions, axis = 1)[:, -5:]
        top5_accuracy = np.mean([targets[i] in top5_preds[i] for i in range(len(targets))]) * 100

        # Per class accuracy: for each class, compute the percentage accuracy
        # (or sensitivity) of model predictions relative to ground truth. Set
        # the accuracy to 0 if the class does not show up in ground truth.
        per_class_acc = []
        for class_id in range(num_classes):
            class_mask = targets == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.mean(pred_classes[class_mask] == class_id) * 100
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)

        metrics = {"top1_accuracy": accuracy_score,
                   "top5_accuracy": top5_accuracy,
                   "per_class_accuracy": per_class_acc,
                   "mean_per_class_accuracy": np.mean(per_class_acc),
                   "num_samples": len(targets)}

        return metrics

def print_training_summary(results):
    """
    Print a formatted summary of training results, passed as a dictionary.
    """
    print("\n")
    print("Training Results Summary")
    print("-" * 24)

    if "best_val_accuracy" in results:
        print(f"Best Validation Accuracy: {results["best_val_accuracy"]:.2f}%")

    if "final_val_accuracy" in results:
        print(f"Final Validation Accuracy: {results["final_val_accuracy"]:.2f}%")

    if "training_time" in results:
        print(f"Training Time: {results["training_time"]:.2f} seconds")

    if "epochs" in results:
        print(f"Total Epochs: {results["epochs"]}")

    # Print improvement over baseline, if available
    if "improvement" in results:
        print(f"Improvement over Baseline: +{results["improvement"]:.2f}%")

    print("-" * 24)

def compare_experiments(baseline_results, augmented_results):
    """
    Compare baseline and augmented experiments by computing absolute and relative
    improvements in best validation accuracy.
    
    Parameters
    ----------
    baseline_results : dict
        Dictionary containing results from the baseline experiment.
    augmented_results : dict
        Dictionary containing results from the augmented experiment.
    
    Returns
    -------
    comparison : dict
        Dictionary with comparison metrics.
    """
    comparison = {}
    
    if "best_val_accuracy" in baseline_results and "best_val_accuracy" in augmented_results:
        # Retrieve best validation accuracies from baseline and augmented expe-
        # riments.
        baseline_acc = baseline_results["best_val_accuracy"]
        augmented_acc = augmented_results["best_val_accuracy"]
        
        # Compute absolute and relative improvements.
        improvement = augmented_acc - baseline_acc
        relative_improvement = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        # Store computed metrics in output dictionary.
        comparison = {"baseline_accuracy": baseline_acc,
                      "augmented_accuracy": augmented_acc,
                      "absolute_improvement": improvement,
                      "relative_improvement": relative_improvement}

        # Display comparison results.
        print(f"\nExperiment Comparison:")
        print(f"Baseline: {baseline_acc:.2f}%")
        print(f"Augmented: {augmented_acc:.2f}%")
        print(f"Improvement: +{improvement:.2f}% ({relative_improvement:.1f}% relative)")
    
    return comparison