import torch
import onnxruntime
import cv2
import torchvision.transforms as torchvision_T
import numpy as np

mean=(0.4611, 0.4359, 0.3905)
std=(0.2193, 0.2150, 0.2109)
common_transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )

image_path = "Epay_AI/IMG_20231214_104950.jpg"
image = cv2.imread(image_path)
h, w = image.shape[:2]
IMAGE_SIZE = 384
# IMAGE_SIZE = image_size
half = IMAGE_SIZE // 2

imH, imW, C = image.shape

image_model = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

scale_x = imW / IMAGE_SIZE
scale_y = imH / IMAGE_SIZE


image_model = common_transforms(image_model)
print(image_model.shape)



sess = onnxruntime.InferenceSession('./wrap.onnx',providers=['CPUExecutionProvider'] )

inputs_name = [x.name for x in sess.get_inputs()]
outputs_name = [x.name for x in sess.get_outputs()]
print(inputs_name)
print(outputs_name)

inputs_shape = [x.shape for x in sess.get_inputs()]
outputs_shape = [x.shape for x in sess.get_outputs()]
print(inputs_shape)
print(outputs_shape)