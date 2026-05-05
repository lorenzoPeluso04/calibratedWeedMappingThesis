import torch
import evaluate
from tqdm.notebook import tqdm
import torch.nn.functional as F

from calweed.train import CustomLoss

from torch import nn


# Define Matrix scaling Calibration
class MatrixScaling(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.C = num_classes
        
        self.P = nn.Parameter(torch.ones( (self.C, self.C) ))  # Learnable P parameters
        self.b = nn.Parameter(torch.ones( (self.C, 1) ))  # Learnable b parameter


    def forward(self, logits):
        B, C, H, W = logits.shape
        # FLATTING

        # Flatten the matrices of each channel and of each batch
        reshape1 = logits.reshape(logits.shape[0], C, -1) # shape = (16, 3, 262'144)

        # Permute dimensions to (C, B, W)
        reshape1_permutated = reshape1.permute(1, 0, 2) # shape = (3, 16, 262'144)

        # Concatenate all different batch vectors for each channel
        Z = reshape1_permutated.reshape(C, -1) # shape (3, 4'194'304)

        calibrated_logits = self.P.to(logits.device)@Z + self.b.to(logits.device) # Scale the logits

        # THICKENING

        # For each channel, the rows of the matrix belong to different batch
        reshape1_permutated_BACK = calibrated_logits.reshape(C, logits.shape[0], -1) # shape = (3, 16, 262'144)

        # Permute dimensions to (B, C, W)
        reshape1_BACK = reshape1_permutated_BACK.permute(1, 0, 2) # shape = (16, 3, 262'144)

        return reshape1_BACK.reshape(logits.shape[0], C, H, W)
    
    
def apply_matrix_scaling(logits, parameters):
    B = logits.shape[0]
    C = logits.shape[1]
    H = logits.shape[2]
    W = logits.shape[3]

    P = torch.Tensor(parameters[0])
    b = torch.Tensor(parameters[1])

    # FLATTING
    # Flatten the matrices of each channel and of each batch
    reshape1 = logits.reshape(B, C, -1) # shape = (16, 3, 262'144)
    # Permute dimensions to (C, B, W)
    reshape1_permutated = reshape1.permute(1, 0, 2) # shape = (3, 16, 262'144)
    # Concatenate all different batch vectors for each channel
    Z = reshape1_permutated.reshape(C, -1) # shape (3, 4'194'304)
    calibrated_logits = P@Z + b # Scale the logits
    # THICKENING
    # For each channel, the rows of the matrix belong to different batch
    reshape1_permutated_BACK = calibrated_logits.reshape(C, B, -1) # shape = (3, 16, 262'144)
    # Permute dimensions to (B, C, W)
    reshape1_BACK = reshape1_permutated_BACK.permute(1, 0, 2) # shape = (16, 3, 262'144)
    return reshape1_BACK.reshape(B, C, H, W)


class TemperatureScaling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))  # Learnable temperature parameter

    def forward(self, logits):
        # print(self.temperature.item()) # -------------------------------------------DEBUG
        return logits / self.temperature.to(logits.device)  # Scale the logits
    
    
def apply_temperature_scaling(logits, parameters):
    temperature = torch.Tensor(parameters[0])
    return logits/temperature


def get_calibration_tecnique(calibration_tecnique_name, cal_params):
    techniques = {
        "temperature_scaling": (TemperatureScaling(**cal_params), apply_temperature_scaling),
        "matrix_scaling": (MatrixScaling(**cal_params), apply_matrix_scaling),
    }
    if calibration_tecnique_name not in techniques:
        raise ValueError(f"Calibration technique '{calibration_tecnique_name}' not recognized.")
    return techniques.get(calibration_tecnique_name)


def finetune_model(model, eval_dataloader, num_epochs=15, calibration_tecnique=None):
    id2label = eval_dataloader.dataset.id2class

    # Define evaluation metrics
    iou_metric = evaluate.load("mean_iou")
    f1_metric = evaluate.load("f1")

    # Define the custom loss function
    custom_loss_fn = CustomLoss(calibration_tecnique=calibration_tecnique)

    # Print parameters -----------------------------------------DEBUG

    for name, param in custom_loss_fn.named_parameters():
        print(name, param.shape, param.requires_grad)

    # Define optimizer
    optimizer = torch.optim.AdamW(custom_loss_fn.parameters(), lr=0.006)

    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set to training mode
    model.train()

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(eval_dataloader)):
            # get the inputs;
            pixel_values = batch["image"].to(device)
            labels = batch["target"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits

            loss, calibrated_logits = custom_loss_fn(logits, labels)

            loss.backward()

            optimizer.step()

            # evaluation for training
            with torch.no_grad():

                softmaxes = F.softmax(calibrated_logits, dim=1)

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

                    print("Loss:", loss.item())
                    print("Mean_iou:", iou_metrics["mean_iou"])
                    print("Mean accuracy:", iou_metrics["mean_accuracy"])
                    print("Mean f1:", f1_metrics["f1"])

    return [p for p in custom_loss_fn.parameters()]
