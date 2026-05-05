import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calweed.model import get_model

# Definizione delle zone e dosaggi
ZONES = {
    "Zona Rossa": {"percentage": 1.0, "dosage": 1.5},  # 100%
    "Zona Blu": {"percentage": 0.8, "dosage": 1.27},   # 80%
    "Zona Gialla": {"percentage": 0.7, "dosage": 1.05}, # 70%
    "Zona Verde": {"percentage": 0.5, "dosage": 0.75},  # 50%
}

# Soglie per assegnare le zone basate sulla percentuale di copertura weed (0-1)
THRESHOLDS = [
    (0.8, "Zona Rossa"),
    (0.6, "Zona Blu"),
    (0.4, "Zona Gialla"),
    (0.0, "Zona Verde"),
]

class HerbicideRecommendationSystem:
    def __init__(self, model_name, id2label, checkpoint_path=None, accuracy=0.8):
        self.model_name = model_name
        self.id2label = id2label
        self.accuracy = accuracy  # Accuracy del modello, da fornire o calcolare
        self.model = get_model(model_name, id2label)
        if checkpoint_path:
            weights = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(weights)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard per ImageNet
        ])

    def predict(self, image):
        """
        Effettua inferenza sull'immagine.
        image: PIL Image o numpy array
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        input_tensor = self.transform(image).unsqueeze(0)  # Aggiungi batch dim
        with torch.no_grad():
            outputs = self.model(pixel_values=input_tensor)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).squeeze(0)  # Shape: [H, W]
        return preds

    def calculate_weed_coverage(self, preds):
        """
        Calcola la percentuale di area coperta da weed.
        preds: torch tensor di predizioni [H, W]
        """
        weed_class = [k for k, v in self.id2label.items() if v == "weed"][0]
        weed_pixels = (preds == weed_class).sum().item()
        total_pixels = preds.numel()
        coverage = weed_pixels / total_pixels
        return coverage

    def apply_tolerance(self, coverage, tolerance_mode='conservative'):
        """
        Applica tolleranza basata sull'accuracy del modello.
        tolerance_mode: 'conservative' (meno erbicida), 'liberal' (più erbicida)
        """
        if tolerance_mode == 'conservative':
            adjusted_coverage = coverage * (1 - self.accuracy)
        elif tolerance_mode == 'liberal':
            adjusted_coverage = coverage * (1 + (1 - self.accuracy))
        else:
            adjusted_coverage = coverage
        return max(0, min(1, adjusted_coverage))  # Clamp tra 0 e 1

    def assign_zone(self, coverage):
        """
        Assegna la zona basata sulla copertura.
        """
        for threshold, zone in THRESHOLDS:
            if coverage >= threshold:
                return zone
        return "Zona Verde"

    def calculate_herbicide_usage(self, zone, area_ha=1.0):
        """
        Calcola l'uso di erbicida per la zona.
        area_ha: area in ettari (default 1 per ha)
        """
        dosage_per_ha = ZONES[zone]["dosage"]
        total_product = dosage_per_ha * area_ha
        return total_product

    def recommend(self, image, tolerance_mode='conservative', area_ha=1.0):
        """
        Sistema completo di raccomandazione.
        """
        preds = self.predict(image)
        coverage = self.calculate_weed_coverage(preds)
        adjusted_coverage = self.apply_tolerance(coverage, tolerance_mode)
        zone = self.assign_zone(adjusted_coverage)
        usage = self.calculate_herbicide_usage(zone, area_ha)
        return {
            "coverage": coverage,
            "adjusted_coverage": adjusted_coverage,
            "zone": zone,
            "herbicide_usage_L": usage,
            "dosage_per_ha": ZONES[zone]["dosage"]
        }

# Esempio di utilizzo
if __name__ == "__main__":
    # Assumi id2label dal dataset
    id2label = {0: "background", 1: "crop", 2: "weed"}
    
    # Crea il sistema
    system = HerbicideRecommendationSystem(
        model_name="segformer",
        id2label=id2label,
        checkpoint_path="weights/segformer.pth",
        accuracy=0.85  # Da calcolare o fornire
    )
    
    # Carica un'immagine di esempio (sostituisci con path reale)
    # image = Image.open("path/to/patch.png")
    # result = system.recommend(image, tolerance_mode='conservative', area_ha=0.01)  # Per una patch piccola
    # print(result)