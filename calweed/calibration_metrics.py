import numpy as np
import torch
from typing import Tuple, Dict, List
from sklearn.metrics import confusion_matrix


def expected_calibration_error_superpixel(superpixel_labels: np.ndarray, 
                                          weed_probs: np.ndarray, 
                                          ground_truth: np.ndarray, 
                                          weed_id: int,
                                          n_bins: int = 10) -> Tuple[float, List[float], List[float]]:
    """
    Calcola ECE per superpixel.
    
    Args:
        superpixel_labels: mappa di label dei superpixel [H, W]
        weed_probs: probabilità weed da modello [H, W]
        ground_truth: ground truth dei pixel [H, W]
        weed_id: indice della classe weed
        n_bins: numero di bin di confidenza
    
    Returns:
        ece, accuracy_per_bin, confidence_per_bin
    """
    max_label = int(superpixel_labels.max())
    superpixel_mean_probs = []
    superpixel_true_weed = []
    
    for label in range(1, max_label + 1):
        mask = superpixel_labels == label
        if not mask.any():
            continue
        
        mean_prob = float(weed_probs[mask].mean())
        true_weed_ratio = float((ground_truth[mask] == weed_id).sum() / mask.sum())
        
        superpixel_mean_probs.append(mean_prob)
        superpixel_true_weed.append(true_weed_ratio)
    
    superpixel_mean_probs = np.array(superpixel_mean_probs)
    superpixel_true_weed = np.array(superpixel_true_weed)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    accuracy_per_bin = []
    confidence_per_bin = []
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        in_bin = (superpixel_mean_probs >= lower) & (superpixel_mean_probs < upper)
        if i == n_bins - 1:
            in_bin = (superpixel_mean_probs >= lower) & (superpixel_mean_probs <= upper)
        
        if in_bin.sum() == 0:
            accuracy_per_bin.append(0.0)
            confidence_per_bin.append(0.0)
            continue
        
        accuracy = float(superpixel_true_weed[in_bin].mean())
        confidence = float(superpixel_mean_probs[in_bin].mean())
        
        accuracy_per_bin.append(accuracy)
        confidence_per_bin.append(confidence)
        
        weight = float(in_bin.sum() / len(superpixel_mean_probs))
        ece += weight * abs(accuracy - confidence)
    
    return ece, accuracy_per_bin, confidence_per_bin


def approximation_quality_spatial_absolute(superpixel_labels: np.ndarray,
                                           weed_preds: np.ndarray,
                                           ground_truth: np.ndarray,
                                           weed_id: int) -> float:
    """
    Calcola AQ spaziale assoluta (somma errori assoluti per superpixel).
    
    AQ = sum(|P_s - G_s|) dove:
    - P_s = pixel weed predetti nel superpixel s
    - G_s = pixel weed reali nel superpixel s
    
    Returns:
        AQ value (pixel count difference)
    """
    max_label = int(superpixel_labels.max())
    aq_total = 0.0
    
    for label in range(1, max_label + 1):
        mask = superpixel_labels == label
        if not mask.any():
            continue
        
        pred_weed_count = float((weed_preds[mask] == weed_id).sum())
        true_weed_count = float((ground_truth[mask] == weed_id).sum())
        
        aq_total += abs(pred_weed_count - true_weed_count)
    
    return aq_total


def overspreading_rate(superpixel_labels: np.ndarray,
                       weed_preds: np.ndarray,
                       ground_truth: np.ndarray,
                       weed_id: int) -> float:
    """
    Over-spraying Rate: frazione di area non infestata trattata come weed.
    
    Over-spraying = Area(FP) / Area(¬weed in GT)
    
    Misura spreco economico e danno ambientale.
    """
    non_weed_gt = ground_truth != weed_id
    false_positives = (weed_preds == weed_id) & non_weed_gt
    
    if non_weed_gt.sum() == 0:
        return 0.0
    
    return float(false_positives.sum() / non_weed_gt.sum())


def underspreading_rate(superpixel_labels: np.ndarray,
                        weed_preds: np.ndarray,
                        ground_truth: np.ndarray,
                        weed_id: int) -> float:
    """
    Under-spraying Rate: frazione di area infestata non trattata.
    
    Under-spraying = Area(FN) / Area(weed in GT)
    
    Misura danno biologico alla resa del raccolto.
    """
    weed_gt = ground_truth == weed_id
    false_negatives = (weed_preds != weed_id) & weed_gt
    
    if weed_gt.sum() == 0:
        return 0.0
    
    return float(false_negatives.sum() / weed_gt.sum())


def compute_all_calibration_metrics(superpixel_labels: np.ndarray,
                                    weed_probs: np.ndarray,
                                    weed_preds: np.ndarray,
                                    ground_truth: np.ndarray,
                                    weed_id: int,
                                    n_bins: int = 10) -> Dict:
    """
    Calcola tutte le metriche di calibrazione e qualità spaziale.
    """
    ece, acc_per_bin, conf_per_bin = expected_calibration_error_superpixel(
        superpixel_labels, weed_probs, ground_truth, weed_id, n_bins
    )
    
    aq = approximation_quality_spatial_absolute(
        superpixel_labels, weed_preds, ground_truth, weed_id
    )
    
    over_spray = overspreading_rate(
        superpixel_labels, weed_preds, ground_truth, weed_id
    )
    
    under_spray = underspreading_rate(
        superpixel_labels, weed_preds, ground_truth, weed_id
    )
    
    return {
        "ece": ece,
        "accuracy_per_bin": acc_per_bin,
        "confidence_per_bin": conf_per_bin,
        "aq_spatial_absolute": aq,
        "overspreading_rate": over_spray,
        "underspreading_rate": under_spray,
    }
