import timm
import torch.nn as nn

from transformers import SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead, SegformerConfig
from transformers.modeling_outputs import SemanticSegmenterOutput

def get_segformer_model(id2label, params=None):
    label2id = {v:k for k,v in id2label.items()} # -> {'background':0, 'crop':1, 'weed':2}

    # define model
    segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                            num_labels= len(id2label), # 3
                                                            id2label= id2label,
                                                            label2id= label2id,
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