import random
import click


@click.group()
def cli():
    pass


@cli.command("train")
@click.option("--model", type=str, default="segformer", help="Model name")
@click.option("--num_epochs", type=int, default=30, help="Number of epochs")
@click.option("--loss", type=str, default="cross_entropy", help="Loss function - cross_entropy or focal")
@click.option("--gamma", type=float, default=1.0, help="Gamma for focal loss")
def train(model, num_epochs, loss, gamma):
    """
    Train the model
    """
    import os
    
    from calweed.data import get_data
    from calweed.model import get_model
    from calweed.train import train_model, save_model_weights

    outfolder = f"experiments/train/{model}_{num_epochs}epochs_{loss}"
    if loss == "focal":
        outfolder += f"_gamma{gamma}"

    # Get data
    train_dataloader, eval_dataloader, test_dataloader = get_data()

    # Get model
    id2label = train_dataloader.dataset.id2class
    torch_model = get_model(model, id2label)

    # Train model
    train_model(torch_model, train_dataloader, outfolder, num_epochs=num_epochs, loss=loss, gamma=gamma)

    # Save model weights
    model_name = f"{model}.pth"
    if loss == "focal":
        model_name = f"{model}_focal_gamma{gamma}.pth"
    save_model_weights(torch_model, folder_path="weights", model_name=model_name)


@cli.command("calibrate")
@click.option("--model", type=str, default="segformer", help="Model name")
@click.option(
    "--calibration_tecnique",
    type=str,
    default="temperature_scaling",
    help="Calibration technique",
)
@click.option(
    "--checkpoint", type=str, default="weights/segformer.pth", help="Checkpoint path"
)
@click.option(
    "--num_epochs", type=int, default=15, help="Number of epochs for calibration"
)
def calibrate(model, calibration_tecnique, num_epochs, checkpoint):
    """
    Calibrate the model
    """
    import torch
    import pickle
    import os

    from calweed.data import get_data
    from calweed.model import get_model
    from calweed.calibrate import finetune_model, get_calibration_tecnique

    # Get data
    _, eval_dataloader, _ = get_data()

    # Get model
    id2label = eval_dataloader.dataset.id2class
    torch_model = get_model(model, id2label)
    weights = torch.load(checkpoint, map_location="cpu")
    torch_model.load_state_dict(weights)
    
    cal_params = {
        "num_classes": len(id2label),
    }
    
    # Get calibration technique
    calibration_tecnique_method, _ = get_calibration_tecnique(calibration_tecnique, cal_params)

    # Calibrate model
    cal_params = finetune_model(torch_model, eval_dataloader, num_epochs, calibration_tecnique_method)

    checkpoint_name = checkpoint.split('/')[-1].split('.')[0]
    with open(os.path.join("weights", f"{model}_calibrated_n{num_epochs}_{calibration_tecnique}_ckpt_{checkpoint_name}.pkl"), 'wb') as f:
        pickle.dump(cal_params, f)


@cli.command("evaluate")
@click.option("--model", type=str, default="segformer", help="Model name")
@click.option(
    "--checkpoint", type=str, default="weights/segformer.pth", help="Checkpoint path"
)
@click.option(
    "--calibration_tecnique",
    type=str,
    default=None,
    help="Calibration technique",
)
@click.option(
    "--calibration_params",
    type=str,
    default=None,
    help="Calibration parameters path",
)
def evaluate(model, checkpoint, calibration_tecnique, calibration_params):
    """
    Evaluate the model
    """
    import torch
    import os
    import pickle
    
    from calweed.data import get_data
    from calweed.model import get_model
    from calweed.evaluate import make_predictions, print_F1_score
    from calweed.metrics import expected_calibration_error, static_calibration_error, show_reliability_diagram
    from calweed.calibrate import get_calibration_tecnique
    

    # Get data
    _, _, test_dataloader = get_data()
    id2label = test_dataloader.dataset.id2class
    
    outfolder = f"experiments/test/{model}_{checkpoint.split('/')[-1].split('.')[0]}"
    if calibration_tecnique is not None:
        cal_params = {
            "num_classes": len(id2label),
        }
        outfolder += f"_{calibration_tecnique}"
        outfolder += f"_{calibration_params.split('/')[-1].split('.')[0]}"
        _, calibrate_fn = get_calibration_tecnique(calibration_tecnique, cal_params)
        with open(calibration_params, 'rb') as f:
            cal_params = pickle.load(f)
    else:
        calibrate_fn = None
        cal_params = None
            
    os.makedirs(outfolder, exist_ok=True)

    # Get model
    torch_model = get_model(model, id2label)
    weights = torch.load(checkpoint, map_location="cpu")
    torch_model.load_state_dict(weights)

    logits, predicted_segmentation_map, labels = make_predictions(torch_model, test_dataloader, calibrate_fn, cal_params)

    f1_metrics = print_F1_score(predicted_segmentation_map, labels, id2label)

    N_BINS = 10

    ece, accuracy_in_bin_list, avg_confidence_in_bin_list = expected_calibration_error(
        logits, predicted_segmentation_map, labels, N_BINS
    )
    print(f"ECE -> {ece}")
    print(f"Accuracy for each bin -> {accuracy_in_bin_list}")
    print(f"Confidence for each bin -> {avg_confidence_in_bin_list}")

    sce, sce_for_class_list = static_calibration_error(
                                                        logits,
                                                        labels,
                                                        n_bins= N_BINS
                                                    )

    print(f"Static calibration Error -> {sce}")
    
    plot = show_reliability_diagram(
        accuracy_bins=accuracy_in_bin_list,
        ece=ece,
    )
    plot.savefig(f"{outfolder}/reliability_diagram.svg")
    
    with open(f"{outfolder}/metrics.txt", "w") as f:
        f.write(f"ECE -> {ece}\n")
        f.write(f"Accuracy for each bin -> {accuracy_in_bin_list}\n")
        f.write(f"Confidence for each bin -> {avg_confidence_in_bin_list}\n")
        f.write(f"SCE -> {sce}\n")
        f.write(f"SCE for each class -> {sce_for_class_list}\n")
        f.write(f"F1 score -> {f1_metrics['f1']}\n")

if __name__ == "__main__":
    cli()
