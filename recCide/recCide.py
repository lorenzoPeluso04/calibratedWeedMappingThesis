import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calweed.model import get_model

standard_dosage_per_ha = 1.5  # Litri per ettaro per copertura totale (100%)

# Definizione delle zone e dosaggi
ZONES = {
    "Zona Rossa": {"percentage": 1.0, "dosage": standard_dosage_per_ha*1.0},  # 100%
    "Zona Blu": {"percentage": 0.8, "dosage": standard_dosage_per_ha*0.8},   # 80%
    "Zona Gialla": {"percentage": 0.7, "dosage": standard_dosage_per_ha*0.7}, # 70%
    "Zona Verde": {"percentage": 0.5, "dosage": standard_dosage_per_ha*0.5},  # 50%
}

# Soglie per assegnare le zone basate sulla percentuale di copertura weed (0-1)
THRESHOLDS = [
    (0.15, "Zona Rossa"),
    (0.10, "Zona Blu"),
    (0.05, "Zona Gialla"),
    (0.00, "Zona Verde"),
]

class HerbicideRecommendationSystem:
    def __init__(self, model_name, id2label, model_variant=None, accuracy=0.8, calibration_file=None):
        """
        Sistema di raccomandazione erbicida.
        
        Args:
            model_name: "segformer" o "mobilenetv4"
            id2label: dizionario classe -> nome
            model_variant: variante del modello (opzionale)
                - None: modello base (es. "segformer.pth")
                - "focal_gamma1.0": calibrato con focal loss gamma=1.0
                - "focal_gamma2.0": calibrato con focal loss gamma=2.0
            accuracy: accuratezza del modello per la tolleranza
            calibration_file: path al file .pkl con parametri di calibrazione (opzionale)
        """
        self.model_name = model_name
        self.id2label = id2label
        self.model_variant = model_variant
        self.accuracy = accuracy  # Accuracy del modello, da fornire o calcolare
        self.calibration_params = None  # Parametri di calibrazione
        
        # Costruisci il path del checkpoint
        if model_variant:
            checkpoint_path = f"weights/{model_name}_{model_variant}.pth"
        else:
            checkpoint_path = f"weights/{model_name}.pth"
        
        self.model = get_model(model_name, id2label)
        if os.path.exists(checkpoint_path):
            weights = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(weights)
            print(f"Caricato modello: {checkpoint_path}")
        else:
            print(f"Attenzione: checkpoint {checkpoint_path} non trovato, uso modello base")
        
        # Carica parametri di calibrazione se forniti
        if calibration_file and os.path.exists(calibration_file):
            with open(calibration_file, 'rb') as f:
                self.calibration_params = pickle.load(f)
            print(f"Caricati parametri di calibrazione da: {calibration_file}")
        elif calibration_file:
            print(f"Attenzione: file di calibrazione {calibration_file} non trovato")
        
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor()
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
            
            # Applica calibrazione se parametri disponibili
            if self.calibration_params is not None:
                # Assumi Temperature Scaling (logits / temperature)
                if 'temperature' in self.calibration_params:
                    temperature = self.calibration_params['temperature']
                    if isinstance(temperature, torch.Tensor):
                        temperature = temperature.to(logits.device)
                    logits = logits / temperature
                # Se ci sono altri tipi di calibrazione, aggiungerli qui
                
            preds = torch.argmax(logits, dim=1).squeeze(0)  # Shape: [H, W]
        return preds

    def save_segmented_image(self, preds, output_path):
        """
        Salva l'immagine segmentata come RGB.
        preds: torch tensor [H, W]
        output_path: path dove salvare l'immagine
        """
        # Definire la mappa colori per le classi
        color_map = {
            0: [0, 0, 0],      # background: nero
            1: [0, 255, 0],    # crop: verde
            2: [255, 0, 0],    # weed: rosso
        }
        
        # Creare immagine RGB
        h, w = preds.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in color_map.items():
            mask = (preds == class_id).cpu().numpy()
            rgb_image[mask] = color
        
        pil_image = Image.fromarray(rgb_image)
        pil_image.save(output_path)
        print(f"Immagine segmentata salvata in: {output_path}")

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
            adjusted_coverage = coverage * self.accuracy
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

    def recommend(self, image, tolerance_mode='conservative', area_ha=1.0, output_dir=None):
        """
        Sistema completo di raccomandazione.
        """
        preds = self.predict(image)
        
        # Salva immagine segmentata se output_dir è fornito
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Usa il nome del file originale se disponibile, altrimenti un nome generico
            base_name = getattr(image, 'filename', 'segmented.png')
            if base_name == 'segmented.png':
                output_path = os.path.join(output_dir, base_name)
            else:
                name = os.path.splitext(os.path.basename(base_name))[0] + '_segmented.png'
                output_path = os.path.join(output_dir, name)
            self.save_segmented_image(preds, output_path)
        
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
    
    # Opzioni modello disponibili:
    # - model_variant=None: modello base (segformer.pth)
    # - model_variant="focal_gamma1.0": calibrato con focal loss gamma=1.0
    # - model_variant="focal_gamma2.0": calibrato con focal loss gamma=2.0
    # - calibration_file: path al file .pkl per calibrazione (es. temperature scaling)
    
    # Crea il sistema con modello calibrato (focal loss gamma 2.0 + temperature scaling)
    system = HerbicideRecommendationSystem(
        model_name="segformer",
        id2label=id2label,
        accuracy=0.80,  # Da calcolare o fornire
        calibration_file="weights/segformer_calibrated_n30_temperature_scaling.pkl"  # File .pkl opzionale
    )
    
    # Carica un'immagine di esempio (sostituisci con path reale)
    image = Image.open("RoWeeder/dataset/patches/512/003/RGB/17.png")
    result = system.recommend(image, tolerance_mode='conservative', area_ha=0.01, output_dir="segmented_outputs")  # Per una patch piccola
    print(result)