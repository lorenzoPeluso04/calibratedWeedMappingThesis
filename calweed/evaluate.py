from torch import nn
import torch
import torch.nn.functional as F
from tqdm import tqdm

import evaluate


def make_predictions(model, test_dataloader, calibrate_fn=None, parameters=None):

    # move model to GP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set to evaluation mode
    model.eval()
    
    all_logits = []
    all_labels = []

    # Dataloader returns a single batch of all test images
    for idx, batch in enumerate(tqdm(test_dataloader)):
        # print("Batch:", idx)

        # get the inputs
        pixel_values = batch["image"].to(device)

        # get target
        labels = batch["target"].to(device)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values).logits
            
        all_logits.append(outputs.cpu())
        all_labels.append(labels.cpu())
            

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    """
    This dimensions is a PROBLEM, so:
    1. Reshape the logits tensor shape to be the same as the input one.
    2. Calibrate the logits (if there is a tecnique)
    3. Apply a softmax to the logits.
    4. Compact channels by classifying each pixel
    """

    # Scale width-height of prediction tensor to the shape of ground truth tensor
    upsampled_logits = nn.functional.interpolate(
        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
    )
    # print(F.softmax(upsampled_logits, dim=1)[0,:, 347,347]) #-------------------------------- DEBUG

    if calibrate_fn is not None:
        # Apply the scaling
        upsampled_logits = calibrate_fn(upsampled_logits, parameters)

    # Get the class probability for each pixel
    probs = F.softmax(upsampled_logits, dim=1)
    # print(probs[0,:, 347,347]) #-------------------------------- DEBUG

    # Choose the most probable class for each pixel
    predicted_segmentation_map = probs.argmax(dim=1)
    print(f"Shape of predicted_segmentation_map -> {predicted_segmentation_map.shape}")

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
