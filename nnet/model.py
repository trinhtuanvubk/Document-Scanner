import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_resnet101 


def prepare_model(args):

    # Initialize model with pre-trained weights.
    weights = 'DEFAULT'
    # mbv3_w = "/home/aittgp/vutt/workspace/Document-Scanner/model_repository/model_mbv3_iou_mix_2C049.pth"
    if args.backbone_model == "mbv3":
        model = deeplabv3_mobilenet_v3_large(weights=weights)
    elif args.backbone_model == "r50":
        model = deeplabv3_resnet50(weights=weights)
    elif args.backbone_model == "r101":
        model = deeplabv3_resnet101(weights=weights)
    else:
        raise ValueError("Wrong backbone model passed. Must be one of 'mbv3', 'r50' and 'r101' ")

    # Update the number of output channels for the output layer.
    # This will remove the pre-trained weights for the last layer.
    model.classifier[4]     = nn.LazyConv2d(num_classes, 1)
    model.aux_classifier[4] = nn.LazyConv2d(num_classes, 1)
    
    if args.pretrained_path != None:
        checkpoints = torch.load(args.pretrained_path, map_location=args.device)
        model.load_state_dict(checkpoints, strict=False)
    return model