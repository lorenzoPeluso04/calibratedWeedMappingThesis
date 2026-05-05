import os
import torch
import evaluate
import pandas as pd
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        """
        Unified Focal Loss class for multi-class classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor([gamma]))
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Forward pass to compute the Focal Loss.
        :param logits: Predictions (logits) from the model.
                       Shape:
                         - multi-class: (batch_size, num_classes, height, width)
        :param labels: Ground truth labels.
                        Shape:
                         - multi-class: (batch_size, height, width)
        """
        
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        # print(f"ce_loss shape -> {ce_loss.shape}") # ------------------------------DEBUG

        # Get the class probabilities for each pixel
        pt = torch.exp(-ce_loss)
        # print(f"pt shape -> {pt.shape}") # ------------------------------DEBUG
        
        focal_loss = torch.pow((1 - pt), self.gamma.to(logits.device)) * ce_loss

        # return reducted focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
    
        # return whole focal_loss
        return focal_loss


# Define a Custom Loss Function with calibration technique (if the tecnique is None
# then it'll perform a normal cross-entropy loss)
class CustomLoss(nn.Module):
    def __init__(self, base_loss="cross_entropy", calibration_tecnique=None, gamma=1.0):
        super().__init__()
        self.calibrationTecnique = calibration_tecnique
        loss_dict = {
            "cross_entropy": nn.CrossEntropyLoss(
                ignore_index=255
            ),  # Ignore padding label if used
            "focal": FocalLoss(gamma=gamma, reduction='sum'),
        }
        if base_loss not in loss_dict:
            raise ValueError(f"Loss function '{base_loss}' not recognized.")
        self.loss_fn = loss_dict[base_loss]
    def forward(self, logits, labels):

        # Scale width-height of prediction tensor to the shape of ground truth tensor
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        if self.calibrationTecnique is not None:
            calibrated_logits = self.calibrationTecnique(
                upsampled_logits
            )  # Apply calibration
        else:
            calibrated_logits = upsampled_logits

        return self.loss_fn(calibrated_logits, labels), calibrated_logits
    

# This function saves the model weights in a local folder
def save_model_weights(model, folder_path=".", model_name="model_x.pth"):

    model_path = os.path.join(folder_path, model_name)

    # Create the folder if doesn't exists
    os.makedirs(folder_path, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)

    print(f"✅ Model weights saved to {model_path}")


def train_model(model, train_dataloader, outfolder, num_epochs=30, calibrationTecnique=None, loss="cross_entropy", gamma=1.0):
    id2label = train_dataloader.dataset.id2class

    # Define evaluation metrics
    iou_metric = evaluate.load("mean_iou")
    f1_metric = evaluate.load("f1")

    # Define the custom loss function
    custom_loss_fn = CustomLoss(base_loss=loss, calibration_tecnique=calibrationTecnique)

    # Define optimizer
    if loss == "cross_entropy":
        parameters = model.parameters()
    elif loss == "focal":
        parameters = [ {'params': model.parameters()}, {'params': custom_loss_fn.parameters()} ]
    optimizer = torch.optim.AdamW(parameters, lr=0.00006)

    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set to training mode
    model.train()
    losses = []
    ious = []
    f1s = []
    accs = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            pixel_values = batch["image"].to(device)
            labels = batch["target"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits

            # Compute custom loss
            loss_value, _ = custom_loss_fn(logits, labels)

            loss_value.backward()
            optimizer.step()

            # evaluation for training
            with torch.no_grad():
                # Scale width-height of prediction tensor to the shape of ground truth tensor
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )

                softmaxes = F.softmax(upsampled_logits, dim=1)

                # Choose the most probable class for each pixel
                predicted = softmaxes.argmax(dim=1)

                # note that the metric expects predictions + labels as numpy arrays
                iou_metric.add_batch(
                    predictions=predicted.detach().cpu().numpy(),
                    references=labels.detach().cpu().numpy(),
                )

            # let's print loss and metrics every 100 batches (we have only 18 batch for epoch, so... print them only one)
            if idx % 100 == 0:

                # IoU metric
                iou_metrics = iou_metric._compute(
                    predictions=predicted.cpu(),
                    references=labels.cpu(),
                    num_labels=len(id2label),
                    ignore_index=255,
                    reduce_labels=False,  # we've already reduced the labels ourselves
                )

                # F1 metric
                f1_metrics = f1_metric._compute(
                    predictions=predicted.cpu().flatten(),  # Flatten tensors for classification metrics
                    references=labels.cpu().flatten(),
                    average="macro",
                )

                print("Loss:", loss_value.item())
                print("Mean_iou:", iou_metrics["mean_iou"])
                print("Mean accuracy:", iou_metrics["mean_accuracy"])
                print("Mean f1:", f1_metrics["f1"])
                losses.append(loss_value.item())
                ious.append(iou_metrics["mean_iou"])
                f1s.append(f1_metrics["f1"])
                accs.append(iou_metrics["mean_accuracy"])
                
        # Save model weights
        save_model_weights(
            model, folder_path=f"{outfolder}/checkpoints", model_name=f"model_epoch_{epoch}.pth"
        )
        
    result_df = pd.DataFrame(
        {
            "loss": losses,
            "mean_iou": ious,
            "mean_accuracy": accs,
            "mean_f1": f1s,
        }
    )
    if loss == "focal":
        with open(f"{outfolder}/focal_loss_gamma_{gamma}.txt", "w") as f:
            f.write(f"Focal Loss input gamma -> {gamma}\n")
            f.write(f"Focal Loss output gamma -> {custom_loss_fn.loss_fn.gamma.item()}\n")
    result_df.to_csv(f"{outfolder}/metrics.csv", index=False)
    print(f"✅ Metrics saved to {outfolder}/metrics.csv")
    
