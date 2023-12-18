import os
import gc
import io
import cv2
import base64
import pathlib
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import torch
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large

class wrapModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mean = (0.4611, 0.4359, 0.3905)
        self.std=(0.2193, 0.2150, 0.2109)
        self.transform = torchvision_T.Compose(
                            [
                                # torchvision_T.ToTensor(),
                                torchvision_T.Normalize(self.mean, self.std),
                            ]
        )
                        
    def forward(self, image):
        input_image = self.transform(image)
        out = self.model(input_image)
        out = out["out"]
        out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].squeeze()
        out = out.to(torch.int32)
        return out
        

def load_model(num_classes=2, model_name="mbv3", device=torch.device("cpu")):
    if model_name == "mbv3":
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
        checkpoint_path = os.path.join(os.getcwd(), "model_repository", "model_mbv3_iou_mix_2C049.pth")
    else:
        model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
        checkpoint_path = os.path.join(os.getcwd(), "model_repository", "model_r50_iou_mix_2C020.pth")

    model.to(device)
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()

    _ = model(torch.randn((1, 3, 384, 384)))

    return model

def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    common_transforms = torchvision_T.Compose(
        [
            # torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )
    return common_transforms


if __name__ == "__main__":
    data_path = "./Epay_AI"
    output_path = "./Epay_Output"
    os.makedirs(output_path, exist_ok=True)
    IMAGE_SIZE = 384
    # model = load_model(model_name="mbv3")
    preprocess_transforms = image_preprocess_transforms()
    model = load_model(model_name="r50")
    wrap = wrapModel(model)

    # convert onnx
    image_path = "Epay_AI/IMG_20231214_104950.jpg"
    image = cv2.imread(image_path)
    print(type(image))
    half = IMAGE_SIZE // 2

    imH, imW, C = image.shape

    image_model = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE
    print(image_model.shape)
    print(type(image_model))
    image_model = torch.Tensor(image_model.transpose((2,0,1)))/255.0

    # print(image_model.shape)

    # image_model = preprocess_transforms(image_model)
    print(image_model.shape)
    image_model = torch.unsqueeze(image_model, dim=0)



    torch.onnx.export(wrap, 
                 image_model, 
                 "wrap_with_preprocess.onnx", 
                 input_names=['input'],
                 output_names=['output'],
                 verbose=True
    )
    
