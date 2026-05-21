import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.segmentation import slic, mark_boundaries
from typing import Dict, List
import os

from calweed.recommendation import assign_zone, calculate_herbicide_usage


def compute_superpixels(image_rgb: Image.Image, num_segments: int = 200, compactness: float = 10.0, sigma: float = 1.0):
    image_np = np.asarray(image_rgb.convert("RGB"))
    labels = slic(
        image_np,
        n_segments=num_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=1,
    )
    return labels


def pixels_to_area_m2(num_pixels: int, gsd_m: float) -> float:
    return float(num_pixels) * (gsd_m ** 2)


def pixels_to_area_ha(num_pixels: int, gsd_m: float) -> float:
    return pixels_to_area_m2(num_pixels, gsd_m) / 10000.0


def superpixel_statistics(superpixel_labels: np.ndarray, weed_probs: np.ndarray, gsd_m: float,
                           severity_thresholds: Dict[float, str] = None) -> List[Dict]:
    if severity_thresholds is None:
        # default zone thresholds: same used in recommendation assign_zone
        severity_thresholds = {
            0.10: "Zona Rossa",
            0.07: "Zona Blu",
            0.03: "Zona Gialla",
            0.00: "Zona Verde",
        }

    max_label = int(superpixel_labels.max())
    stats = []
    for label in range(1, max_label + 1):
        mask = superpixel_labels == label
        if not mask.any():
            continue
        mean_prob = float(weed_probs[mask].mean())
        pixel_count = int(mask.sum())
        area_m2 = pixels_to_area_m2(pixel_count, gsd_m)
        area_ha = pixels_to_area_ha(pixel_count, gsd_m)
        zone = assign_zone(mean_prob)
        usage_l = calculate_herbicide_usage(zone, area_ha)
        stats.append({
            "label": label,
            "mean_weed_prob": mean_prob,
            "pixel_count": pixel_count,
            "area_m2": area_m2,
            "area_ha": area_ha,
            "zone": zone,
            "usage_L": usage_l,
        })
    return stats


def visualize_superpixels(superpixel_labels: np.ndarray, image_rgb: Image.Image = None) -> Image.Image:
    """
    Visualizza i superpixel tracciando i loro bordi in cyan sovrapposti all'immagine RGB.
    
    Args:
        superpixel_labels: matrice dei label dei superpixel (output di compute_superpixels)
        image_rgb: immagine RGB originale (richiesta per la visualizzazione)
    
    Returns:
        PIL Image con bordi superpixel sovrapposti
    """
    if image_rgb is None:
        raise ValueError("image_rgb è richiesta per visualizzare i bordi dei superpixel")
    
    # Converti immagine PIL a numpy array
    image_np = np.asarray(image_rgb)
    
    # Normalizza i valori tra 0 e 1 per mark_boundaries
    if image_np.max() > 1:
        image_normalized = image_np / 255.0
    else:
        image_normalized = image_np
    
    # Traccia i bordi dei superpixel (color=cyan [0, 1, 1])
    segmented_image = mark_boundaries(image_normalized, superpixel_labels, color=(0, 1, 1), mode='outer')
    
    # Converti a PIL Image (mark_boundaries ritorna valori 0-1)
    segmented_pil = Image.fromarray((segmented_image * 255).astype(np.uint8), mode="RGB")
    
    return segmented_pil

    # --- VECCHIO CODICE (Colori casuali per ogni superpixel) ---
    # Decommentare le seguenti righe per ottenere i colori randomici:
    # max_label = int(superpixel_labels.max())
    # # Crea una colormap con numero sufficiente di colori
    # np.random.seed(42)
    # colors = np.random.randint(0, 256, size=(max_label + 1, 3))
    # # Mappa i label ai colori
    # segmented_image = colors[superpixel_labels]
    # # Converti a PIL Image
    # segmented_pil = Image.fromarray(segmented_image.astype(np.uint8), mode="RGB")
    # return segmented_pil


def save_superpixel_segmentation(superpixel_labels: np.ndarray, image_rgb: Image.Image, 
                                 output_dir: str, filename: str = None, image_name: str = None):
    """
    Salva l'immagine dei superpixel segmentati in una cartella apposita.
    
    Args:
        superpixel_labels: matrice dei label dei superpixel
        image_rgb: immagine RGB originale
        output_dir: cartella principale dove salvare
        filename: nome del file (default: costruito da image_name)
        image_name: nome/identificativo dell'immagine per il filename
    """
    segmented_folder = os.path.join(output_dir, "segmented")
    os.makedirs(segmented_folder, exist_ok=True)
    
    if filename is None:
        if image_name:
            base_name = os.path.splitext(image_name)[0]
            filename = f"{base_name}_superpixel_segmentation.png"
        else:
            filename = "superpixel_segmentation.png"
    
    segmented_image = visualize_superpixels(superpixel_labels, image_rgb)
    output_path = os.path.join(segmented_folder, filename)
    segmented_image.save(output_path)
    
    return output_path
