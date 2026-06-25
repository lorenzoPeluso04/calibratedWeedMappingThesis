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


def get_superpixel_states(superpixel_labels: np.ndarray, weed_probs: np.ndarray, threshold: float) -> Dict[int, bool]:
    """
    Determina lo stato di ogni superpixel (infestato=True, libero=False) basato sulla soglia.
    
    Args:
        superpixel_labels: matrice dei label dei superpixel
        weed_probs: matrice delle probabilità di erbaccia per pixel
        threshold: soglia per determinare se un superpixel è infestato
    
    Returns:
        Dizionario {label: is_treated} dove True = infestato, False = libero
    """
    max_label = int(superpixel_labels.max())
    states = {}
    
    for label in range(1, max_label + 1):
        mask = superpixel_labels == label
        if not mask.any():
            continue
        mean_prob = float(weed_probs[mask].mean())
        states[label] = mean_prob >= threshold
    
    return states


def get_adjacent_superpixels(superpixel_labels: np.ndarray) -> Dict[int, set]:
    """
    Calcola la mappa di adiacenza tra superpixel (4-connessione).
    
    Args:
        superpixel_labels: matrice dei label dei superpixel
    
    Returns:
        Dizionario {label: set di label adiacenti}
    """
    h, w = superpixel_labels.shape
    adjacency = {}

    max_label = int(superpixel_labels.max())
    for label in range(1, max_label + 1):
        adjacency[label] = set()

    # Adiacenza verticale (vettorizzata)
    vertical_diff = superpixel_labels[:-1, :] != superpixel_labels[1:, :]
    v1 = superpixel_labels[:-1, :][vertical_diff]
    v2 = superpixel_labels[1:, :][vertical_diff]
    for a, b in zip(v1.tolist(), v2.tolist()):
        adjacency[a].add(b)
        adjacency[b].add(a)

    # Adiacenza orizzontale (vettorizzata)
    horizontal_diff = superpixel_labels[:, :-1] != superpixel_labels[:, 1:]
    h1 = superpixel_labels[:, :-1][horizontal_diff]
    h2 = superpixel_labels[:, 1:][horizontal_diff]
    for a, b in zip(h1.tolist(), h2.tolist()):
        adjacency[a].add(b)
        adjacency[b].add(a)

    return adjacency


def merge_adjacent_superpixels(superpixel_labels: np.ndarray, states: Dict[int, bool]) -> np.ndarray:
    """
    Unisce superpixel adiacenti che hanno lo stesso stato (infestato/libero).
    
    Args:
        superpixel_labels: matrice dei label dei superpixel
        states: dizionario {label: is_treated} dello stato di ogni superpixel
    
    Returns:
        Nuova matrice di label con superpixel uniti
    """
    adjacency = get_adjacent_superpixels(superpixel_labels)
    
    # Union-Find per merge
    parent = {}
    max_label = int(superpixel_labels.max())
    
    for label in range(1, max_label + 1):
        parent[label] = label
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Merge superpixel adiacenti con lo stesso stato
    for label in range(1, max_label + 1):
        for adjacent_label in adjacency.get(label, set()):
            if states.get(label) == states.get(adjacent_label):
                union(label, adjacent_label)
    
    # Crea una mappa di rimapping (old_label -> new_label)
    remap = {}
    new_label_counter = 1
    for label in range(1, max_label + 1):
        root = find(label)
        if root not in remap:
            remap[root] = new_label_counter
            new_label_counter += 1
        remap[label] = remap[root]
    
    # Applica il remapping
    merged_labels = np.zeros_like(superpixel_labels)
    for label in range(1, max_label + 1):
        mask = superpixel_labels == label
        merged_labels[mask] = remap[label]
    
    return merged_labels


def visualize_superpixels_by_state(superpixel_labels: np.ndarray, weed_probs: np.ndarray, 
                                    threshold: float, image_rgb: Image.Image = None) -> Image.Image:
    """
    Visualizza i superpixel colorati in base al loro stato (infestato/libero).
    
    Args:
        superpixel_labels: matrice dei label dei superpixel
        weed_probs: matrice delle probabilità di erbaccia
        threshold: soglia per determinare se un superpixel è infestato
        image_rgb: immagine RGB originale (opzionale, per overlay)
    
    Returns:
        PIL Image con superpixel colorati per stato
    """
    states = get_superpixel_states(superpixel_labels, weed_probs, threshold)
    max_label = int(superpixel_labels.max())
    
    # Crea immagine colorata
    h, w = superpixel_labels.shape
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label in range(1, max_label + 1):
        mask = superpixel_labels == label
        if not mask.any():
            continue
        # Rosso per infestato, Verde per libero
        color = [255, 0, 0] if states.get(label, False) else [0, 255, 0]
        colored_image[mask] = color
    
    # Aggiungi bordi se disponibile immagine originale
    if image_rgb is not None:
        image_np = np.asarray(image_rgb)
        if image_np.max() > 1:
            image_normalized = image_np / 255.0
        else:
            image_normalized = image_np
        
        # mark_boundaries ritorna immagine normalizzata (0-1)
        segmented = mark_boundaries(image_normalized, superpixel_labels, color=(1, 1, 1), mode='outer')
        colored_float = colored_image.astype(np.float32) / 255.0
        
        # Blend: 70% colore, 30% bordi
        blended = 0.7 * colored_float + 0.3 * segmented
        colored_image = (blended * 255).astype(np.uint8)
    
    return Image.fromarray(colored_image, mode="RGB")


def visualize_superpixels_by_prob(superpixel_labels: np.ndarray, weed_probs: np.ndarray,
                                  image_rgb: Image.Image = None,
                                  colormap_name: str = 'YlOrRd') -> Image.Image:
    """
    Visualizza i superpixel colorati in funzione della probabilità media di infestante (\bar{P}_i),
    usando una scala di calore continua (giallo=bassa prob, rosso=alta prob).

    Args:
        superpixel_labels: matrice dei label dei superpixel
        weed_probs: matrice delle probabilità per-pixel della classe weed
        image_rgb: immagine RGB originale (opzionale, per overlay dei bordi)
        colormap_name: nome della colormap matplotlib (default 'YlOrRd')

    Returns:
        PIL Image con superpixel colorati per probabilità media
    """
    max_label = int(superpixel_labels.max())
    h, w = superpixel_labels.shape
    prob_map = np.zeros((h, w), dtype=np.float32)

    for label in range(1, max_label + 1):
        mask = superpixel_labels == label
        if not mask.any():
            continue
        prob_map[mask] = float(weed_probs[mask].mean())

    colormap = plt.colormaps[colormap_name]
    colored_rgba = colormap(prob_map)          # shape (H, W, 4), valori in [0, 1]
    colored_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)

    if image_rgb is not None:
        image_np = np.asarray(image_rgb)
        image_normalized = image_np / 255.0 if image_np.max() > 1 else image_np.astype(float)
        borders = mark_boundaries(image_normalized, superpixel_labels, color=(1, 1, 1), mode='outer')
        blended = 0.75 * (colored_rgb.astype(np.float32) / 255.0) + 0.25 * borders
        colored_rgb = (blended * 255).astype(np.uint8)

    result = Image.fromarray(colored_rgb, mode='RGB')

    # Disegna la probabilità media come testo al centroide di ogni superpixel
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(9, h // 60))
    except Exception:
        font = ImageFont.load_default()

    for label in range(1, max_label + 1):
        mask = superpixel_labels == label
        if not mask.any():
            continue
        mean_prob = float(weed_probs[mask].mean())
        # Centroide del superpixel
        rows, cols = np.where(mask)
        cy, cx = int(rows.mean()), int(cols.mean())
        text = f"{mean_prob:.2f}"
        # Colore testo: nero se sfondo chiaro, bianco se scuro
        text_color = (0, 0, 0) if mean_prob < 0.5 else (255, 255, 255)
        draw.text((cx, cy), text, fill=text_color, font=font, anchor="mm")

    return result


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
