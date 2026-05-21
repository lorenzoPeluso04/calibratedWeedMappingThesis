import timm
import torch.nn as nn

from transformers import SegformerForSemanticSegmentation, SegformerConfig
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
from transformers.modeling_outputs import SemanticSegmenterOutput

def get_segformer_model(id2label, params=None):
    label2id = {v:k for k,v in id2label.items()} # -> {'background':0, 'crop':1, 'weed':2}

    # load the backbone config, then override labels to match the downstream task
    config = SegformerConfig.from_pretrained("nvidia/mit-b0")
    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id

    segformer_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        config=config,
        ignore_mismatched_sizes=True,
    )
    return segformer_model


class MobileNetV4Former(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV4Former, self).__init__()
        self.encoder = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k',
            pretrained=True,
            features_only=True,
        )
        
        config = SegformerConfig(hidden_sizes=[32, 32, 64, 96, 960], num_labels=num_classes)
        self.head = SegformerDecodeHead(config)
        
    def forward(self, pixel_values, labels=None):
        x = self.encoder(pixel_values)
        x = self.head(x)
        return SemanticSegmenterOutput(logits=x, loss=None)
    
    
def get_mobilenetv4_model(id2label, params=None):
    # define model
    mobilenetv4_model = MobileNetV4Former(num_classes=len(id2label))
    return mobilenetv4_model


model_dict = {
    "segformer": get_segformer_model,
    "mobilenetv4": get_mobilenetv4_model,
}


def get_model(model_name: str, id2label, params=None):
    """
    Get the model from the model_dict
    """
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found in model_dict")
    return model_dict[model_name](id2label, params)


def load_segmentation_model(model_name: str, weights=None, device="cpu"):
    """
    Load a segmentation model with pre-trained weights.
    
    Args:
        model_name: Name of the model ('segformer' or 'mobilenetv4')
        weights: Path to saved model weights
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded model on the specified device
    """
    import torch
    
    # Default id2label mapping for RoWeeder dataset
    id2label = {0: "background", 1: "crop", 2: "weed"}
    
    # Get the base model
    model = get_model(model_name, id2label)
    
    # Load weights if provided
    if weights is not None:
        if isinstance(weights, str):
            checkpoint = torch.load(weights, map_location=device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    return model