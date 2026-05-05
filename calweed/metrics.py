import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def expected_calibration_error(logits, predicted_labels, true_labels, M=5):

    accuracy_in_bin_list = []
    avg_confidence_in_bin_list = []

    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Get the max probability for each pixel
    probs = F.softmax(logits, dim=1)
    confidences = torch.max(probs, dim=1)[0].flatten()

    # Predictions
    predicted_labels = predicted_labels.flatten()

    # Ground-truth
    true_labels = true_labels.flatten()

    # get a boolean Tensor of correct/false predictions
    accuracies = predicted_labels.to(logits.device) == true_labels.to(logits.device)

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(
            confidences > bin_lower.item(), confidences <= bin_upper.item()
        ).bool()
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.float().mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].float().mean().item()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

            accuracy_in_bin_list.append(accuracy_in_bin)
            avg_confidence_in_bin_list.append(avg_confidence_in_bin)

    return ece.item(), accuracy_in_bin_list, avg_confidence_in_bin_list


def show_reliability_diagram(accuracy_bins, ece=None, show_title=True, show_xlabel=True, show_ylabel=True):

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))

    # ---------------------------------------------------------------LINE---------
    # Draw the bisector of the first and third quadrants
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, color="black", linestyle="--", alpha=0.5)

    # Create 10 bins between 0 and 1. There are 11 edges for 10 bins.
    confidence_bins = np.arange(0, 11) / 10

    # Calculate bin centers for plotting the bars
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2

    # -------------------------------------------------------------BAR CHART 1----
    # Reshape accuracy_bins to the correct length
    achieved_accuracy = np.zeros(10)
    achieved_accuracy[-len(accuracy_bins) :] = accuracy_bins

    # Draw bars representing the accuracy achieved by the bins
    bar1 = ax.bar(
        bin_centers,
        achieved_accuracy,
        width=0.1,
        edgecolor="blue",
        alpha=0.5,
        align="center",
    )

    # -------------------------------------------------------------BAR CHART 2----
    # the right accuracy for each bin
    balanced_accuracy = np.arange(1, 11) / 10
    balanced_accuracy[: (10 - len(accuracy_bins))] = 0
    balanced_accuracy -= achieved_accuracy

    bar2 = ax.bar(
        bin_centers,
        balanced_accuracy,
        bottom=achieved_accuracy,
        width=0.1,
        facecolor="none",
        edgecolor="red",
        align="center",
    )

    if show_title:
        ax.set_title("Reliability Diagram")
    if show_xlabel:
        ax.set_xlabel("Confidence")
    if show_ylabel:
        ax.set_ylabel("Accuracy")
    ax.set_xticks(confidence_bins)
    ax.legend(
        [bar1, bar2],
        [
            "Outputs",
            "Gap/Surplus",
        ],
    )

    ax.text(
        0.35,
        0.9,
        f"ECE= {ece:.3f}",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

    return fig

def static_calibration_error(logits, labels, num_classes=3, n_bins=10):
    """
    Computes Static Calibration Error (SCE) for a batch of segmentation outputs.

    Args:
        logits: Tensor of shape [B, C, H, W]
        labels: Tensor of shape [B, H, W]
        num_classes: number of classes
        n_bins: number of bins to compute calibration in

    Returns:
        sce: scalar tensor, the static calibration error
    """

    label2id = {'background':0, 'crop':1, 'weed':2}
    
    sce = 0.0 
    sce_for_class = {'background': 0, 'crop': 0, 'weed': 0}
    
    B, C, H, W = logits.shape

    probs = F.softmax(logits.to(logits.device), dim=1)           # [B, C, H, W]
    confs, preds = torch.max(probs, dim=1)     # [B, H, W]
    
    for key, cls in label2id.items():
        # Create mask for where ground truth == current class
        cls_mask = (labels == cls).to(logits.device)             # [B, H, W]
        
        # Probabilities for class 'cls' at each pixel
        cls_probs = probs[:, cls, :, :]        # [B, H, W]
        
        # Correct predictions for that class
        cls_correct = (preds == cls) & cls_mask   # [B, H, W]
        
        # Mask out irrelevant values
        cls_probs = cls_probs[cls_mask]        # [N] for class cls
        cls_correct = cls_correct[cls_mask].float()  # [N] correctness
        
        if cls_probs.numel() == 0:
            continue  # no pixels of this class in batch

        # Binning
        bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1, device=logits.device)
        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (cls_probs >= low) & (cls_probs < high)

            if in_bin.sum() > 0:
                avg_conf = cls_probs[in_bin].mean()
                acc = cls_correct[in_bin].mean()
                
                sce += torch.abs(avg_conf - acc) * in_bin.float().mean()
                sce_for_class[key] += torch.abs(avg_conf - acc) * in_bin.float().mean()

    return sce / num_classes, sce_for_class