import random
import click
import numpy as np

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


@cli.command("infer-superpixels")
@click.option("--model", type=str, default="segformer", help="Model name")
@click.option("--model-variant", type=str, default=None, help="Model variant for weights filename")
@click.option("--checkpoint", type=str, default=None, help="Checkpoint path")
@click.option("--calibration-file", type=str, default=None, help="Calibration parameters .pkl file")
@click.option("--image", type=str, required=True, help="Path to input RGB image")
@click.option("--output-dir", type=str, default="segmented_outputs", help="Directory to save results")
@click.option("--num-segments", type=int, default=200, help="Number of superpixels")
@click.option("--compactness", type=float, default=10.0, help="SLIC compactness")
@click.option("--sigma", type=float, default=1.0, help="SLIC smoothing sigma")
@click.option("--gsd-m", type=float, default=0.05, help="Ground Sampling Distance in meters/pixel")
@click.option("--tolerance-mode", type=str, default="conservative", help="Tolerance mode for coverage adjustment")
@click.option("--accuracy", type=float, default=0.99, help="Model accuracy for tolerance adjustment")
def infer_superpixels(model, model_variant, checkpoint, calibration_file, image, output_dir, num_segments, compactness, sigma, gsd_m, tolerance_mode, accuracy):
    """Esegui inferenza superpixel ed effettua la raccomandazione erbicida."""
    from recCide import HerbicideRecommendationSystem

    system = HerbicideRecommendationSystem(
        model_name=model,
        id2label={0: "background", 1: "crop", 2: "weed"},
        model_variant=model_variant,
        accuracy=accuracy,
        calibration_file=calibration_file,
        checkpoint_path=checkpoint,
    )

    result = system.recommend_superpixels(
        image,
        num_segments=num_segments,
        compactness=compactness,
        sigma=sigma,
        tolerance_mode=tolerance_mode,
        gsd_m=gsd_m,
        output_dir=output_dir,
    )

    print("\n--- Risultato superpixel ---")
    print(f"Superpixel totali: {result['n_superpixels']}")
    print(f"Area totale stimata (ha): {result['total_area_ha']:.4f}")
    print(f"Erbicida totale stimato (L): {result['total_herbicide_usage_L']:.4f}")
    print(f"Report salvato in: {output_dir}/superpixel_recommendation.csv")


@cli.command("benchmark-superpixels")
@click.option("--image-dir", type=str, default=None, help="Path to directory with RGB images")
@click.option("--ground-truth-dir", type=str, default=None, help="Path to directory with ground truth images")
@click.option("--image", type=str, default=None, help="Path to single RGB image (alternative to --image-dir)")
@click.option("--ground-truth", type=str, default=None, help="Path to single ground truth (alternative to --ground-truth-dir)")
@click.option("--output-dir", type=str, default="benchmark_results", help="Output directory for results")
@click.option("--num-segments", type=int, default=200, help="Number of superpixels")
@click.option("--gsd-m", type=float, default=0.05, help="Ground Sampling Distance in meters/pixel")
@click.option("--accuracy", type=float, default=0.8, help="Model accuracy")
def benchmark_superpixels(image_dir, ground_truth_dir, image, ground_truth, output_dir, num_segments, gsd_m, accuracy):
    """Esegui benchmark comparativo con THRESHOLD SWEEP: modello base vs calibrato vs focal."""
    import os
    import glob
    import numpy as np
    from pathlib import Path
    from recCide import HerbicideRecommendationSystem

    id2label = {0: "background", 1: "crop", 2: "weed"}
    
    models_to_test = [
        {"name": "segformer", "variant": None, "label": "Base (non calibrato)"},
        {"name": "segformer", "variant": None, "label": "Temp Scaling", "calibration": "weights/segformer_calibrated_n30_temperature_scaling.pkl"},
        {"name": "segformer", "variant": "focal_gamma1.0", "label": "Focal Loss γ=1.0"},
        {"name": "segformer", "variant": "focal_gamma2.0", "label": "Focal Loss γ=2.0"},
        {"name": "segformer", "variant": "focal_gamma2.0", "label": "Focal + Temp Scaling", "calibration": "weights/segformer_calibrated_n30_temperature_scaling.pkl"},
    ]
    
    # Definiamo il range di soglie (tau) per misurare la libertà dell'agricoltore
    thresholds_to_sweep = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    os.makedirs(output_dir, exist_ok=True)
    
    if image and ground_truth:
        image_gt_pairs = [(image, ground_truth)]
        print(f"Modalità: singola immagine")
    elif image_dir and ground_truth_dir:
        if not os.path.isdir(image_dir) or not os.path.isdir(ground_truth_dir):
            print(f"Errore: le directory non esistono")
            return
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg")))
        if not image_files:
            print(f"Errore: nessuna immagine trovata in {image_dir}")
            return
        image_gt_pairs = []
        for img_path in image_files:
            img_name = Path(img_path).stem
            gt_path = os.path.join(ground_truth_dir, f"{img_name}.png")
            if os.path.exists(gt_path):
                image_gt_pairs.append((img_path, gt_path))
            else:
                print(f"Avviso: ground truth non trovato per {img_path}")
        print(f"Modalità: cartella con {len(image_gt_pairs)} coppie immagine-ground_truth")
    else:
        print("Errore: fornire --image e --ground-truth oppure --image-dir e --ground-truth-dir")
        return
    
    all_results_summary = []
    
    for img_idx, (img_path, gt_path) in enumerate(image_gt_pairs, 1):
        image_base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n{'='*60}")
        print(f"Immagine {img_idx}/{len(image_gt_pairs)}: {image_base_name}")
        print(f"{'='*60}")
        
        results_summary = []
        
        for config in models_to_test:
            model_name = config["name"]
            variant = config["variant"]
            label = config["label"]
            calibration_file = config.get("calibration", None)
            
            print(f"\n=== Testing: {label} ===")
            
            system = HerbicideRecommendationSystem(
                model_name=model_name,
                id2label=id2label,
                model_variant=variant,
                accuracy=accuracy,
                calibration_file=calibration_file,
            )
            
            # Eseguiamo lo sweep su tutte le soglie per questo modello
            for tau in thresholds_to_sweep:
                print(f"  Valutazione con Soglia \u03c4 = {tau:.2f}...")
                
                eval_subdir = os.path.join(output_dir, image_base_name, label.replace(" ", "_").replace("=", ""), f"tau_{tau:.2f}")
                
                # NOTA: Assicurati che il metodo evaluate_superpixels in recCide.py 
                # accetti il parametro 'threshold' e usi la nuova logica che abbiamo scritto!
                metrics = system.evaluate_superpixels(
                    img_path,
                    gt_path,
                    num_segments=num_segments,
                    gsd_m=gsd_m,
                    output_dir=eval_subdir,
                    threshold=tau, 
                )
                
                results_summary.append({
                    "image": image_base_name,
                    "model": label,
                    "threshold": tau,
                    "ece": metrics["ece"],
                    "aq_spatial_absolute": metrics["aq_spatial_absolute"],
                    "overspreading_rate": metrics["overspreading_rate"],
                    "underspreading_rate": metrics["underspreading_rate"],
                })
                all_results_summary.append(results_summary[-1])
        
        # Salva il riepilogo per questa immagine includendo la colonna Threshold
        summary_path = os.path.join(output_dir, f"benchmark_summary_merged_{image_base_name}_{num_segments}.csv")
        with open(summary_path, "w") as f:
            headers = ["Model", "Threshold", "ECE", "AQ_Spatial_Absolute", "Over_spraying", "Under_spraying"]
            f.write(",".join(headers) + "\n")
            for row in results_summary:
                values = [
                    row["model"],
                    f"{row['threshold']:.2f}",
                    f"{row['ece']:.4f}",
                    f"{row['aq_spatial_absolute']:.2f}",
                    f"{row['overspreading_rate']:.4f}",
                    f"{row['underspreading_rate']:.4f}",
                ]
                f.write(",".join(values) + "\n")
        
        print(f"\nRiepilogo con Sweep per {image_base_name} salvato in: {summary_path}")
    
    # Salva il riepilogo aggregato se ci sono più immagini
    if len(image_gt_pairs) > 1:
        aggregate_summary_path = os.path.join(output_dir, f"benchmark_summary_aggregate_merged_{num_segments}.csv")
        with open(aggregate_summary_path, "w") as f:
            headers = ["Image", "Model", "Threshold", "ECE", "AQ_Spatial_Absolute", "Over_spraying", "Under_spraying"]
            f.write(",".join(headers) + "\n")
            for row in all_results_summary:
                values = [
                    str(row["image"]),
                    str(row["model"]),
                    f"{row['threshold']:.2f}",
                    f"{row['ece']:.4f}",  # <-- CORRETTO: Ora scrive il valore dell'ECE, non il nome del modello!
                    f"{row['aq_spatial_absolute']:.2f}",
                    f"{row['overspreading_rate']:.4f}",
                    f"{row['underspreading_rate']:.4f}",
                ]
                f.write(",".join(values) + "\n")
        print(f"\nRiepilogo aggregato salvato in: {aggregate_summary_path}")

    print(f"\n{'='*60}\n=== Benchmark completato ===\n{'='*60}")


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
    from calweed.evaluate import make_predictions, print_F1_score, compute_herbicide_saving_metrics
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

    saving_metrics = compute_herbicide_saving_metrics(predicted_segmentation_map, labels, id2label, area_ha=1.0)
    print("\n--- Herbicide savings evaluation ---")
    print(f"Mean herbicide saving index -> {saving_metrics['mean_saving_index']:.4f}")
    print(f"Median herbicide saving index -> {saving_metrics['median_saving_index']:.4f}")
    print(f"Mean weed coverage underestimation -> {saving_metrics['mean_weed_coverage_underestimation']:.4f}")
    print(f"Zone counts -> {saving_metrics['zone_counts']}")
    print(f"Average saving per zone -> {saving_metrics['zone_savings']}")

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
        f.write(f"Mean herbicide saving index -> {saving_metrics['mean_saving_index']:.4f}\n")
        f.write(f"Median herbicide saving index -> {saving_metrics['median_saving_index']:.4f}\n")
        f.write(f"Mean weed coverage underestimation -> {saving_metrics['mean_weed_coverage_underestimation']:.4f}\n")
        f.write(f"Zone counts -> {saving_metrics['zone_counts']}\n")
        f.write(f"Average saving per zone -> {saving_metrics['zone_savings']}\n")


if __name__ == "__main__":
    cli()
