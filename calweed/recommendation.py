import torch
from typing import Dict, Tuple, Union

STANDARD_DOSAGE_PER_HA = 1.5

ZONES = {
    "Zona Rossa": {"percentage": 1.0, "dosage": STANDARD_DOSAGE_PER_HA * 1.0},
    "Zona Blu": {"percentage": 0.8, "dosage": STANDARD_DOSAGE_PER_HA * 0.8},
    "Zona Gialla": {"percentage": 0.5, "dosage": STANDARD_DOSAGE_PER_HA * 0.5},
    "Zona Verde": {"percentage": 0.0, "dosage": STANDARD_DOSAGE_PER_HA * 0.0},
}

THRESHOLDS = [
    (0.10, "Zona Rossa"),
    (0.07, "Zona Blu"),
    (0.03, "Zona Gialla"),
    (0.00, "Zona Verde"),
]


def assign_zone(coverage: float) -> str:
    for threshold, zone in THRESHOLDS:
        if coverage >= threshold:
            return zone
    return "Zona Verde"


def calculate_herbicide_usage(zone: str, area_ha: float = 1.0) -> float:
    dosage_per_ha = ZONES[zone]["dosage"]
    return dosage_per_ha * area_ha


def calculate_usage_and_zone(coverage: float, area_ha: float = 1.0) -> Tuple[float, str]:
    zone = assign_zone(coverage)
    usage = calculate_herbicide_usage(zone, area_ha)
    return usage, zone


def calculate_weed_coverage(preds: torch.Tensor, id2label: Dict[int, str] = None, weed_id: int = None) -> float:
    if weed_id is None:
        if id2label is None:
            raise ValueError("Either id2label or weed_id must be provided")
        weed_id = next(k for k, v in id2label.items() if v == "weed")
    weed_pixels = (preds == weed_id).sum().item()
    total_pixels = preds.numel()
    return weed_pixels / total_pixels if total_pixels > 0 else 0.0


def apply_tolerance(coverage: float, accuracy: float, tolerance_mode: str = "conservative") -> float:
    if tolerance_mode == "conservative":
        adjusted = coverage * accuracy
    elif tolerance_mode == "liberal":
        adjusted = coverage * (1 + (1 - accuracy))
    else:
        adjusted = coverage
    return max(0.0, min(1.0, adjusted))


def calculate_herbicide_saving_index(usage: float, area_ha: float = 1.0) -> float:
    baseline = STANDARD_DOSAGE_PER_HA * area_ha
    saving = 1.0 - (usage / baseline)
    return float(saving) if saving >= 0 else 0.0
