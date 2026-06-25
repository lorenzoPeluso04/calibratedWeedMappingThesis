import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import os
import pickle
from typing import Iterable, List, Optional, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calweed.model import get_model
from calweed.recommendation import (
    ZONES,
    assign_zone,
    calculate_herbicide_usage,
    calculate_weed_coverage,
    apply_tolerance,
)
from calweed.superpixel import (
    compute_superpixels,
    superpixel_statistics,
    pixels_to_area_m2,
    pixels_to_area_ha,
    save_superpixel_segmentation,
    get_superpixel_states,
    merge_adjacent_superpixels,
    visualize_superpixels_by_state,
    visualize_superpixels_by_prob,
)

class HerbicideRecommendationSystem:
    # Mappatura threshold → file di calibrazione per valutazione superpixel
    """THRESHOLD_CALIBRATION_MAP = {
        0.10: "weights/segformer_calibrated_n30_temperature_scaling.pkl",
        0.20: "weights/segformer_calibrated_n30_temperature_scaling.pkl",
        0.30: "weights/segformer_calibrated_n30_temperature_scaling.pkl",
        0.40: "weights/segformer_calibrated_n30_temperature_scaling.pkl",
        0.50: "weights/segformer_calibrated_n30_temperature_scaling_ckpt_segformer_focal_gamma2.pkl",
    }"""
    
    THRESHOLD_CALIBRATION_MAP = {
        0.10: None,  # Usa calibrazione di default per tau=0.10
        0.20: None,
        0.30: None,
        0.40: None,
        0.50: None,
    }

    def __init__(self, model_name, id2label, model_variant=None, accuracy=0.9, 
                 calibration_file=None, checkpoint_path=None):
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
            checkpoint_path: path esplicito del file di checkpoint del modello
        """
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_name = model_name
        self.id2label = id2label
        self.model_variant = model_variant
        self.accuracy = accuracy  # Accuracy del modello, da fornire o calcolare
        self.calibration_params = None  # Parametri di calibrazione (default per raccomandazione)
        self.threshold_calibrations = {}  # Cache di calibrazioni per threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path:
            self.checkpoint_path = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(self.root_dir, checkpoint_path)
        elif model_variant:
            self.checkpoint_path = os.path.join(self.root_dir, "weights", f"{model_name}_{model_variant}.pth")
        else:
            self.checkpoint_path = os.path.join(self.root_dir, "weights", f"{model_name}.pth")

        self.model = get_model(model_name, id2label).to(self.device)
        if os.path.exists(self.checkpoint_path):
            weights = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(weights)
            print(f"✓ Caricato modello: {self.checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"❌ Checkpoint non trovato: {self.checkpoint_path}\n"
                f"Assicurati che il file esista in: weights/{model_name}.pth"
            )

        if calibration_file:
            self.load_calibration_params(calibration_file)

        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_calibration_params(self, calibration_file: str):
        calibrated_path = calibration_file
        if not os.path.isabs(calibration_file):
            calibrated_path = os.path.join(self.root_dir, calibration_file)

        if os.path.exists(calibrated_path):
            with open(calibrated_path, 'rb') as f:
                self.calibration_params = pickle.load(f)
            print(f"Caricati parametri di calibrazione da: {calibrated_path}")
        else:
            print(f"Attenzione: file di calibrazione {calibrated_path} non trovato")
            self.calibration_params = None
    
    def load_calibration_for_threshold(self, threshold: float):
        """
        Carica i parametri di calibrazione specifici per un dato threshold.
        Implementa cache per evitare ricaricamenti multipli.
        
        Args:
            threshold: valore del threshold (0.10, 0.20, 0.30, 0.40, 0.50)
            
        Returns:
            Parametri di calibrazione caricati dal file, o None se non trovato
        """
        # Controlla se è già in cache
        if threshold in self.threshold_calibrations:
            return self.threshold_calibrations[threshold]
        
        # Se threshold non è nella mappa, ritorna None
        if threshold not in self.THRESHOLD_CALIBRATION_MAP:
            print(f"⚠️  Threshold {threshold} non ha calibrazione mappata. Usando calibrazione di default.")
            return self.calibration_params
        
        calib_file = self.THRESHOLD_CALIBRATION_MAP[threshold]
        
        # Se non c'è calibrazione specifica per questo threshold, usa quella di default
        if calib_file is None:
            print(f"⚠️  Threshold {threshold} usa calibrazione di default.")
            self.threshold_calibrations[threshold] = self.calibration_params
            return self.calibration_params
        
        calib_path = calib_file if os.path.isabs(calib_file) else os.path.join(self.root_dir, calib_file)
        
        if os.path.exists(calib_path):
            with open(calib_path, 'rb') as f:
                params = pickle.load(f)
            self.threshold_calibrations[threshold] = params
            print(f"✓ Caricata calibrazione per threshold={threshold}: {calib_path}")
            return params
        else:
            print(f"⚠️  Calibrazione non trovata per threshold={threshold}: {calib_path}")
            self.threshold_calibrations[threshold] = None
            return None

    def _apply_calibration(self, logits):
        if self.calibration_params is None:
            return logits

        if isinstance(self.calibration_params, dict):
            if 'temperature' in self.calibration_params:
                temperature = self.calibration_params['temperature']
                if isinstance(temperature, torch.Tensor):
                    temperature = temperature.to(logits.device)
                return logits / temperature
            return logits

        if isinstance(self.calibration_params, (list, tuple)):
            if len(self.calibration_params) == 1:
                temperature = self.calibration_params[0]
                if isinstance(temperature, torch.nn.Parameter):
                    temperature = temperature.detach()
                temperature = temperature.to(logits.device)
                return logits / temperature

            if len(self.calibration_params) == 2:
                P, b = self.calibration_params
                if isinstance(P, torch.nn.Parameter):
                    P = P.detach()
                if isinstance(b, torch.nn.Parameter):
                    b = b.detach()
                P = P.to(logits.device)
                b = b.to(logits.device)
                B, C, H, W = logits.shape
                reshape1 = logits.reshape(B, C, -1)
                reshape1_permutated = reshape1.permute(1, 0, 2)
                Z = reshape1_permutated.reshape(C, -1)
                calibrated_logits = P @ Z + b
                reshape1_permutated_BACK = calibrated_logits.reshape(C, B, -1)
                reshape1_BACK = reshape1_permutated_BACK.permute(1, 0, 2)
                return reshape1_BACK.reshape(B, C, H, W)

        raise ValueError("Formato dei parametri di calibrazione non supportato")

    def predict(self, image: Union[str, np.ndarray, Image.Image], return_probs: bool = True):
        """
        Effettua inferenza sull'immagine.
        image: path, PIL Image o numpy array
        return_probs: se True restituisce anche le probabilità per classe

        Le predizioni sono restituite come tensor di classe per pixel, 
        e opzionalmente anche le probabilità per classe.
        Le predizioni riguardano i singoli pixel, non i superpixel. 
        La raccomandazione finale sarà basata sui superpixel, ma questa 
        funzione restituisce le predizioni a livello di pixel.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(image.astype('uint8'))
            elif image.ndim == 3 and image.shape[0] == 3:
                image = Image.fromarray(np.transpose(image, (1, 2, 0)).astype('uint8'))
            else:
                raise ValueError("Unsupported numpy image shape: {}".format(image.shape))

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=input_tensor)
            logits = outputs.logits
            logits = self._apply_calibration(logits)
            if logits.shape[-2:] != input_tensor.shape[-2:]:
                logits = F.interpolate(logits, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).squeeze(0)

        if return_probs:
            return preds, probs.squeeze(0)
        return preds

    def recommend_from_path(self, image_path: str, tolerance_mode='liberal', area_ha=1.0, output_dir=None):
        return self.recommend(image_path, tolerance_mode=tolerance_mode, area_ha=area_ha, output_dir=output_dir)

    def recommend_superpixels(self, image: Union[str, np.ndarray, Image.Image], num_segments: int = 200,
                              compactness: float = 10.0, sigma: float = 1.0, tolerance_mode: str = 'liberal',
                              gsd_m: float = 0.05, output_dir: Optional[str] = None, 
                              ground_truth_path: Optional[str] = None, threshold: float = 0.20):
        """
        Sistema di raccomandazione superpixel con valutazione e visualizzazione.
        
        Args:
            image: path, PIL Image o numpy array dell'immagine
            num_segments: numero di superpixel
            compactness: compattezza dei superpixel
            sigma: sigma per SLIC
            tolerance_mode: modalità di tolleranza
            gsd_m: Ground Sampling Distance in m/pixel
            output_dir: directory di output
            ground_truth_path: path alla ground truth (opzionale, per valutazione)
            threshold: soglia tau per determinare se un superpixel è infestato
        """
        image_path = None
        if isinstance(image, str):
            image_path = image
            image = Image.open(image).convert("RGB")
        elif hasattr(image, 'filename'):
            image_path = image.filename

        # Inferenza
        preds, probs = self.predict(image, return_probs=True)
        weed_id = next(k for k, v in self.id2label.items() if v == "weed")
        weed_probs = probs[weed_id].cpu().numpy()

        # Superpixel classici
        labels_original = compute_superpixels(image, num_segments=num_segments, compactness=compactness, sigma=sigma)
        stats = superpixel_statistics(labels_original, weed_probs, gsd_m=gsd_m)

        # Calcolo stato superpixel e merge
        states = get_superpixel_states(labels_original, weed_probs, threshold)
        labels_merged = merge_adjacent_superpixels(labels_original, states)

        # Statistiche per superpixel originali e merged
        stats_original = superpixel_statistics(labels_original, weed_probs, gsd_m=gsd_m)
        stats_merged = superpixel_statistics(labels_merged, weed_probs, gsd_m=gsd_m)

        total_usage_original = sum(item["usage_L"] for item in stats_original)
        total_area_ha_original = sum(item["area_ha"] for item in stats_original)
        
        total_usage_merged = sum(item["usage_L"] for item in stats_merged)
        total_area_ha_merged = sum(item["area_ha"] for item in stats_merged)

        result = {
            "n_superpixels_original": len(stats_original),
            "total_area_ha_original": total_area_ha_original,
            "total_herbicide_usage_L_original": total_usage_original,
            "superpixel_stats_original": stats_original,
            
            "n_superpixels_merged": len(stats_merged),
            "total_area_ha_merged": total_area_ha_merged,
            "total_herbicide_usage_L_merged": total_usage_merged,
            "superpixel_stats_merged": stats_merged,
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Estrai il nome dell'immagine per i file di output
            image_name = os.path.basename(image_path) if image_path else "image"
            base_name = os.path.splitext(image_name)[0]
            
            # 0. Salva l'immagine di segmentazione semantica (tutte le classi)
            seg_output_path = os.path.join(output_dir, f"{base_name}_segmented.png")
            self.save_segmented_image(preds, seg_output_path)

            # 0b. Salva la heatmap di probabilità per-pixel (\hat{P}_weed)
            import matplotlib
            heatmap_colored = (matplotlib.colormaps['hot'](weed_probs)[:, :, :3] * 255).astype(np.uint8)
            heatmap_path = os.path.join(output_dir, f"{base_name}_weed_prob_heatmap.png")
            Image.fromarray(heatmap_colored, mode='RGB').save(heatmap_path)
            print(f"✓ Heatmap probabilità per-pixel salvata in: {heatmap_path}")

            # 1. Salva l'immagine superpixel classica con colori per stato
            colored_original = visualize_superpixels_by_state(labels_original, weed_probs, threshold, image)
            colored_original_path = os.path.join(output_dir, f"{base_name}_tau_{threshold}_{num_segments}_superpixel_original_state.png")
            colored_original.save(colored_original_path)
            print(f"✓ Superpixel originali (colorati per stato) salvati in: {colored_original_path}")
            
            # 2. Salva l'immagine superpixel merged con colori per stato
            colored_merged = visualize_superpixels_by_state(labels_merged, weed_probs, threshold, image)
            colored_merged_path = os.path.join(output_dir, f"{base_name}_tau_{threshold}_{num_segments}superpixel_merged_state.png")
            colored_merged.save(colored_merged_path)
            print(f"✓ Superpixel uniti (colorati per stato) salvati in: {colored_merged_path}")
            
            # 2b. Salva superpixel originali colorati per probabilità media (\bar{P}_i)
            prob_original = visualize_superpixels_by_prob(labels_original, weed_probs, image)
            prob_original_path = os.path.join(output_dir, f"{base_name}_tau_{threshold}_{num_segments}_superpixel_original_prob.png")
            prob_original.save(prob_original_path)
            print(f"✓ Superpixel originali (colorati per prob media) salvati in: {prob_original_path}")

            # 2c. Salva superpixel merged colorati per probabilità media (\bar{P}_i)
            prob_merged = visualize_superpixels_by_prob(labels_merged, weed_probs, image)
            prob_merged_path = os.path.join(output_dir, f"{base_name}_tau_{threshold}_{num_segments}_superpixel_merged_prob.png")
            prob_merged.save(prob_merged_path)
            print(f"✓ Superpixel uniti (colorati per prob media) salvati in: {prob_merged_path}")

            # 3. Salva immagine superpixel originale con bordi
            segmented_original = save_superpixel_segmentation(labels_original, image, output_dir, 
                                                             filename=f"{base_name}_tau_{threshold}_{num_segments}_superpixel_original_borders.png")
            print(f"✓ Superpixel originali (con bordi) salvati in: {segmented_original}")
            
            # 4. Salva immagine superpixel merged con bordi
            from skimage.segmentation import mark_boundaries
            image_np = np.asarray(image)
            if image_np.max() > 1:
                image_normalized = image_np / 255.0
            else:
                image_normalized = image_np
            segmented_merged_array = mark_boundaries(image_normalized, labels_merged, color=(0, 1, 1), mode='outer')
            segmented_merged_pil = Image.fromarray((segmented_merged_array * 255).astype(np.uint8), mode="RGB")
            segmented_merged_path = os.path.join(output_dir, f"{base_name}_tau_{threshold}_{num_segments}_superpixel_merged_borders.png")
            segmented_merged_pil.save(segmented_merged_path)
            print(f"✓ Superpixel uniti (con bordi) salvati in: {segmented_merged_path}")
            
            # Salva il report CSV per superpixel originali
            report_original_path = os.path.join(output_dir, f"superpixel_recommendation_original_{base_name}_{self.model_name}.csv")
            with open(report_original_path, "w") as f:
                headers = ["label", "mean_weed_prob", "pixel_count", "area_m2", "area_ha", "zone", "usage_L"]
                f.write(",".join(headers) + "\n")
                for item in stats_original:
                    row = [str(item[h]) for h in headers]
                    f.write(",".join(row) + "\n")
            print(f"✓ Report superpixel originali salvato in: {report_original_path}")
            
            # Salva il report CSV per superpixel merged
            report_merged_path = os.path.join(output_dir, f"superpixel_recommendation_merged_{base_name}_{self.model_name}_{num_segments}.csv")
            with open(report_merged_path, "w") as f:
                headers = ["label", "mean_weed_prob", "pixel_count", "area_m2", "area_ha", "zone", "usage_L"]
                f.write(",".join(headers) + "\n")
                for item in stats_merged:
                    row = [str(item[h]) for h in headers]
                    f.write(",".join(row) + "\n")
            print(f"✓ Report superpixel uniti salvato in: {report_merged_path}")
            
            # Esegui valutazione se ground truth è fornito
            if ground_truth_path:
                print(f"\n📊 Esecuzione valutazione superpixel con threshold={threshold}...")
                evaluation_result = self.evaluate_superpixels(
                    image_path=image_path,
                    ground_truth_path=ground_truth_path,
                    num_segments=num_segments,
                    compactness=compactness,
                    sigma=sigma,
                    gsd_m=gsd_m,
                    output_dir=output_dir,
                    threshold=threshold
                )
                result["evaluation_original"] = evaluation_result


                print(f"📊 Valutazione completata originale:")
                print(f"   ECE: {evaluation_result['ece']:.4f}")
                print(f"   AQ Spatial Absolute: {evaluation_result['aq_spatial_absolute']:.4f}")
                print(f"   Over-spraying rate: {evaluation_result['overspreading_rate']:.4f}")
                print(f"   Under-spraying rate: {evaluation_result['underspreading_rate']:.4f}")

                evaluation_result_merged = self.evaluate_superpixels(
                    image_path=image_path,
                    ground_truth_path=ground_truth_path,
                    num_segments=num_segments,
                    compactness=compactness,
                    sigma=sigma,
                    gsd_m=gsd_m,
                    output_dir=output_dir,
                    threshold=threshold
                )
                result["evaluation_merged"] = evaluation_result_merged
                print(f"📊 Valutazione completata uniti:"
                      f"   ECE: {evaluation_result_merged['ece']:.4f}\n"
                      f"   AQ Spatial Absolute: {evaluation_result_merged['aq_spatial_absolute']:.4f}\n"
                      f"   Over-spraying rate: {evaluation_result_merged['overspreading_rate']:.4f}\n"
                      f"   Under-spraying rate: {evaluation_result_merged['underspreading_rate']:.4f}\n")
                result["evaluation_merged"] = evaluation_result_merged

        return result

    def evaluate_superpixels(self, 
                             image_path: str, 
                             ground_truth_path: str, 
                             num_segments: int = 200, 
                             compactness: float = 10.0, 
                             sigma: float = 1.0, 
                             gsd_m: float = 0.05,
                             output_dir: str = "evaluation_outputs",
                             threshold: float = 0.20) -> dict:
        """
        Valuta le performance spaziali del modello sul campionamento a superpixel
        eseguendo il calcolo dinamico in base alla soglia dell'agricoltore.
        
        La calibrazione cambia dinamicamente in base al threshold fornito.
        """
        import cv2
        from skimage.segmentation import slic
        from calweed.calibration_metrics import expected_calibration_error_superpixel

        os.makedirs(output_dir, exist_ok=True)
        
        # Carica calibrazione specifica per il threshold (una sola volta)
        print(f"\n📊 Caricamento calibrazione per threshold={threshold}...")
        threshold_calibration = self.load_calibration_for_threshold(threshold)
        # Salva la calibrazione corrente e usa quella del threshold
        original_calibration = self.calibration_params
        self.calibration_params = threshold_calibration
        
        # 1. Caricamento immagine e Ground Truth
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Carichiamo la maschera come array numpy in scala di grigi
        gt_raw = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        
        # Creiamo una nuova maschera vuota con gli ID corretti (0 = background)
        gt = np.zeros_like(gt_raw)
        
        # Mappiamo i valori convertiti dall'RGB a scala di grigi
        gt[gt_raw == 149] = 1  # Il verde (Crop) diventa ID 1
        gt[gt_raw == 76] = 2   # Il rosso (Weed) diventa ID 2
            
        weed_id = 2
        
        # 2. Segmentazione in Superpixel dell'immagine RGB originale
        superpixel_labels = slic(
            image_rgb, 
            n_segments=num_segments, 
            compactness=compactness, 
            sigma=sigma, 
            start_label=1
        )
        
        # 3. Inferenza del modello (ottiene le probabilità per pixel)
        pil_img = Image.fromarray(image_rgb)
        preds, probs = self.predict(pil_img, return_probs=True) 
        
        # Trasformiamo il canale della weed in un array numpy compatibile con le metriche
        weed_probs = probs[weed_id].cpu().numpy()
        
        # 3.5 Applicazione logica di unione superpixel
        states = get_superpixel_states(superpixel_labels, weed_probs, threshold)
        superpixel_labels_merged = merge_adjacent_superpixels(superpixel_labels, states)
        
        # 4. Calcolo dell'ECE sui superpixel ORIGINALI (non uniti)
        ece_orig, acc_bin_orig, conf_bin_orig = expected_calibration_error_superpixel(
            superpixel_labels=superpixel_labels,
            weed_probs=weed_probs,
            ground_truth=gt,
            weed_id=weed_id,
            n_bins=10
        )

        # 4b. Calcolo dell'ECE sui superpixel MERGED
        ece_merged, acc_bin, conf_bin = expected_calibration_error_superpixel(
            superpixel_labels=superpixel_labels_merged,
            weed_probs=weed_probs,
            ground_truth=gt,
            weed_id=weed_id,
            n_bins=10
        )
        
        # 5. Calcolo delle metriche spaziali sui superpixel ORIGINALI
        spatial_metrics_orig = evaluate_superpixel_decisions(
            superpixel_labels=superpixel_labels,
            weed_probs=weed_probs,
            ground_truth=gt,
            weed_id=weed_id,
            threshold=threshold
        )

        # 5b. Calcolo delle metriche spaziali sui superpixel MERGED
        spatial_metrics_merged = evaluate_superpixel_decisions(
            superpixel_labels=superpixel_labels_merged,
            weed_probs=weed_probs,
            ground_truth=gt,
            weed_id=weed_id,
            threshold=threshold
        )
        
        # Ripristina la calibrazione originale
        self.calibration_params = original_calibration
        
        # Salvataggio visivo opzionale (puoi tenerlo o rimuoverlo se rallenta lo sweep)
        # save_superpixel_segmentation(image_rgb, superpixel_labels, os.path.join(output_dir, "segmentation.png"))
        
        return {
            # Superpixel originali (non uniti)
            "ece": ece_orig,
            "aq_spatial_absolute": spatial_metrics_orig["aq_spatial_absolute"],
            "overspreading_rate": spatial_metrics_orig["overspreading_rate"],
            "underspreading_rate": spatial_metrics_orig["underspreading_rate"],
            # Superpixel merged
            "ece_merged": ece_merged,
            "aq_spatial_absolute_merged": spatial_metrics_merged["aq_spatial_absolute"],
            "overspreading_rate_merged": spatial_metrics_merged["overspreading_rate"],
            "underspreading_rate_merged": spatial_metrics_merged["underspreading_rate"],
        }

    def recommend_batch(self, image_paths: Iterable[str], tolerance_mode='liberal', area_ha=1.0, output_dir=None):
        results = {}
        for image_path in image_paths:
            results[image_path] = self.recommend_from_path(image_path, tolerance_mode=tolerance_mode, area_ha=area_ha, output_dir=output_dir)
        return results

    def save_segmented_image(self, preds, output_path):
        """
        Salva l'immagine segmentata come RGB.
        preds: torch tensor [H, W] o [1, H, W] o [B, H, W]
        output_path: path dove salvare l'immagine
        """
        # Assicurati che preds sia 2D (H, W)
        if preds.dim() > 2:
            preds = preds.squeeze()
        if preds.dim() > 2:
            preds = preds[0]  # Prendi il primo elemento del batch
        
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
        return calculate_weed_coverage(preds, self.id2label)

    def apply_tolerance(self, coverage, tolerance_mode='conservative'):
        """
        Applica tolleranza basata sull'accuracy del modello.
        tolerance_mode: 'conservative' (meno erbicida), 'liberal' (più erbicida)
        """
        return apply_tolerance(coverage, self.accuracy, tolerance_mode)

    def assign_zone(self, coverage):
        """
        Assegna la zona basata sulla copertura.
        """
        return assign_zone(coverage)

    def calculate_herbicide_usage(self, zone, area_ha=1.0):
        """
        Calcola l'uso di erbicida per la zona.
        area_ha: area in ettari (default 1 per ha)
        """
        return calculate_herbicide_usage(zone, area_ha)

    def recommend(self, image, tolerance_mode='conservative', area_ha=1.0, output_dir=None):
        """
        Sistema completo di raccomandazione.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        preds, probs = self.predict(image, return_probs=True)
        
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

def evaluate_superpixel_decisions(superpixel_labels: np.ndarray,
                                  weed_probs: np.ndarray,
                                  ground_truth: np.ndarray,
                                  weed_id: int,
                                  threshold: float = 0.10) -> dict:
    """
    Calcola AQ, Over-spraying e Under-spraying basandosi sulle decisioni
    prese a livello di superpixel data una specifica soglia operativa (tau).
    """
    max_label = int(superpixel_labels.max())
    
    total_fp_pixels = 0  # Pixel sani spruzzati per errore (Spreco)
    total_fn_pixels = 0  # Pixel infestati NON spruzzati (Danno biologico)
    
    total_clean_pixels_in_gt = (ground_truth != weed_id).sum()
    total_weed_pixels_in_gt = (ground_truth == weed_id).sum()
    
    aq_total = 0.0
    
    for label in range(1, max_label + 1):
        mask = superpixel_labels == label
        if not mask.any():
            continue
        
        # 1. Realtà del Ground Truth nel superpixel
        true_weed_count = (ground_truth[mask] == weed_id).sum()
        superpixel_area = mask.sum()
        
        # 2. Decisione del sistema in base alla probabilità media e alla soglia tau
        mean_prob = float(weed_probs[mask].mean())
        is_treated = mean_prob >= threshold
        
        # 3. Accumulo errori basato sull'impatto reale dei pixel
        if is_treated:
            # Se spruzzi il superpixel, i pixel sani al suo interno sono False Positives
            total_fp_pixels += (superpixel_area - true_weed_count)
            aq_total += abs(superpixel_area - true_weed_count)
        else:
            # Se NON lo spruzzi, i pixel infestati al suo interno sono False Negatives
            total_fn_pixels += true_weed_count
            aq_total += true_weed_count

    # Calcolo dei tassi globali rispetto al totale dei pixel dell'immagine
    over_spray_rate = total_fp_pixels / total_clean_pixels_in_gt if total_clean_pixels_in_gt > 0 else 0.0
    under_spray_rate = total_fn_pixels / total_weed_pixels_in_gt if total_weed_pixels_in_gt > 0 else 0.0
    
    #print(f"DEBUG - Pixel Erbacce nel GT: {total_weed_pixels_in_gt}, Pixel FN (Mancati): {total_fn_pixels}")

    return {
        "aq_spatial_absolute": float(aq_total),
        "overspreading_rate": float(over_spray_rate),
        "underspreading_rate": float(under_spray_rate)
    }
# Esempio di utilizzo
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema di raccomandazione erbicida")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["recommend", "superpixels"],
        default="superpixels",
        help="Modalità di raccomandazione: 'recommend' (pixel) o 'superpixels' (superpixel)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="RoWeeder/dataset/patches/512/003/RGB/14.png",
        help="Path all'immagine di input"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path alla ground truth (solo per valutazione superpixel)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="segmented_outputs",
        help="Directory di output"
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        default=200,
        help="Numero di superpixel (solo per modalità superpixels)"
    )
    parser.add_argument(
        "--gsd-m",
        type=float,
        default=0.05,
        help="Ground Sampling Distance in m/pixel"
    )
    parser.add_argument(
        "--area-ha",
        type=float,
        default=1.0,
        help="Area in ettari"
    )
    parser.add_argument(
        "--tolerance-mode",
        type=str,
        choices=["conservative", "liberal"],
        default="liberal",
        help="Modalità di tolleranza"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        choices=[0.10, 0.20, 0.30, 0.40, 0.50],
        default=0.20,
        help="Soglia operativa (tau) per la valutazione superpixel (solo per modalità superpixels)"
    )
    
    args = parser.parse_args()
    
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
    
    print(f"\n{'='*60}")
    print(f"Modalità: {args.mode.upper()}")
    print(f"Immagine: {args.image}")
    print(f"Directory output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    if args.mode == "superpixels":
        # Raccomandazione con superpixel - passa il path, non la PIL Image
        result = system.recommend_superpixels(
            args.image,  # Pass path directly
            num_segments=args.num_segments,
            gsd_m=args.gsd_m,
            tolerance_mode=args.tolerance_mode,
            output_dir=args.output_dir,
            ground_truth_path=args.ground_truth,
            threshold=args.threshold
        )
        print("\n📊 RISULTATI RACCOMANDAZIONE SUPERPIXEL (ORIGINALI):")
        print(f"  Numero di superpixel: {result['n_superpixels_original']}")
        print(f"  Area totale: {result['total_area_ha_original']:.6f} ha")
        print(f"  Uso totale erbicida: {result['total_herbicide_usage_L_original']:.4f} L")
        
        print("\n📊 RISULTATI RACCOMANDAZIONE SUPERPIXEL (UNITI):")
        print(f"  Numero di superpixel: {result['n_superpixels_merged']}")
        print(f"  Area totale: {result['total_area_ha_merged']:.6f} ha")
        print(f"  Uso totale erbicida: {result['total_herbicide_usage_L_merged']:.4f} L")
        
        if "evaluation" in result:
            print("\n📊 METRICHE DI VALUTAZIONE: (ORIGINALI)")
            print(f"  ECE: {result['evaluation']['ece']:.4f}")
            print(f"  AQ Spatial Absolute: {result['evaluation']['aq_spatial_absolute']:.4f}")
            print(f"  Over-spraying rate: {result['evaluation']['overspreading_rate']:.4f}")
            print(f"  Under-spraying rate: {result['evaluation']['underspreading_rate']:.4f}")

            print("\n📊 METRICHE DI VALUTAZIONE: (UNITI)")
            print(f"  ECE: {result['evaluation']['ece']:.4f}")
            print(f"  AQ Spatial Absolute: {result['evaluation']['aq_spatial_absolute']:.4f}")
            print(f"  Over-spraying rate: {result['evaluation']['overspreading_rate']:.4f}")
            print(f"  Under-spraying rate: {result['evaluation']['underspreading_rate']:.4f}")
        
    else:
        # Raccomandazione standard a livello di pixel
        image = Image.open(args.image)
        result = system.recommend(
            image,
            tolerance_mode=args.tolerance_mode,
            area_ha=args.area_ha,
            output_dir=args.output_dir
        )
        print("🎯 RISULTATI RACCOMANDAZIONE PIXEL:")
        print(f"  Copertura erbacce: {result['coverage']*100:.2f}%")
        print(f"  Copertura aggiustata: {result['adjusted_coverage']*100:.2f}%")
        print(f"  Zona: {result['zone']}")
        print(f"  Uso erbicida: {result['herbicide_usage_L']:.4f} L")
        print(f"  Dosaggio per ha: {result['dosage_per_ha']} L/ha")
    
    print(f"\n✓ Analisi completata!\n")