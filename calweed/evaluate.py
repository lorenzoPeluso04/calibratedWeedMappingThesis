from torch import nn
import torch
import torch.nn.functional as F
from tqdm import tqdm

import evaluate


def make_predictions(model, test_dataloader, calibrate_fn=None, parameters=None):
    # Sposta il modello su GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Imposta in modalità valutazione
    model.eval()
    
    all_logits = []
    all_labels = []

    # Recupera l'ID della classe weed dinamicamente dal dataset
    id2label = test_dataloader.dataset.id2class
    weed_id = next(k for k, v in id2label.items() if v == "weed") # Risulterà 2

    # Fase di raccolta dati
    for idx, batch in enumerate(tqdm(test_dataloader)):
        pixel_values = batch["image"].to(device)
        labels_batch = batch["target"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values).logits
            
        all_logits.append(outputs.cpu())
        all_labels.append(labels_batch.cpu())
            
    # Concatenazione di tutti i batch
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    """
    This dimensions is a PROBLEM, so:
    1. Reshape the logits tensor shape to be the same as the input one.
    2. Calibrate the logits (if there is a tecnique)
    3. Apply a softmax to the logits.
    4. Compact channels by classifying each pixel
    """

    # 1. Upsampling dei logits alla dimensione della ground truth[cite: 3]
    upsampled_logits = nn.functional.interpolate(
        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
    )

    # 2. Eventuale calibrazione[cite: 3]
    if calibrate_fn is not None:
        upsampled_logits = calibrate_fn(upsampled_logits, parameters)

    # 3. Calcolo delle probabilità Softmax[cite: 3, 7]
    probs = F.softmax(upsampled_logits, dim=1)

    # 4. Predizione finale (argmax)[cite: 3, 7]
    predicted_segmentation_map = probs.argmax(dim=1)

    # --- ANALISI DELLE PROBABILITÀ PER LA CLASSE WEED ---
    
    # Estraiamo solo il canale relativo alle erbacce (indice 2)
    weed_probs = probs[:, weed_id, :, :]

    # Maschere booleane
    pred_weed_mask = (predicted_segmentation_map == weed_id)  # Pixel classificati come weed
    true_weed_mask = (labels == weed_id)                    # Pixel che sono realmente weed
    tp_weed_mask = pred_weed_mask & true_weed_mask          # True Positives (classificati correttamente)

    print("\n--- Analisi Confidenza Modello (Classe Weed) ---")
    
    # Media probabilità quando il modello PREVEDE weed
    if pred_weed_mask.sum() > 0:
        avg_prob_pred = weed_probs[pred_weed_mask].mean().item()
        print(f"Probabilità media nei pixel PREDETTI come weed: {avg_prob_pred:.4f}")
    
    # Media probabilità della classe weed nei pixel REALMENTE weed (Ground Truth)
    if true_weed_mask.sum() > 0:
        avg_prob_true = weed_probs[true_weed_mask].mean().item()
        print(f"Probabilità media (canale weed) nei pixel REALMENTE weed: {avg_prob_true:.4f}")

    # Media probabilità nei pixel weed identificati CORRETTAMENTE (True Positives)
    if tp_weed_mask.sum() > 0:
        avg_prob_tp = weed_probs[tp_weed_mask].mean().item()
        print(f"Probabilità media nei True Positives (weed): {avg_prob_tp:.4f}")

    print(f"Shape of predicted_segmentation_map -> {predicted_segmentation_map.shape}\n")

    return upsampled_logits, predicted_segmentation_map, labels


# print the F1 scores
def print_F1_score(predictions, labels, id2label):

    # Define the evaluation metrics
    f1_metric = evaluate.load("f1")

    # metric expects a list of numpy arrays for both predictions and references
    f1_metrics = f1_metric._compute(
        predictions=predictions.detach().cpu().flatten(),
        references=labels.detach().cpu().flatten(),
        average=None,  # This ensures per-class F1 scores are returned
    )

    # Print overall mean F1 score
    print("Mean F1:", f1_metrics["f1"].mean())

    # Print F1 score per class
    for class_id, class_name in id2label.items():
        print(
            f"F1 score for {class_name} (class {class_id}): {f1_metrics['f1'][class_id]}"
        )
        
    return f1_metrics
