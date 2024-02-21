import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import albumentations as A
import PIL

from torchvision.utils import make_grid, save_image
from torchmetrics import MeanMetric
from livelossplot import PlotLosses

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# For reproducibility
seed = 41
seed_everything(seed)


train_loader, valid_loader = get_dataset(DATA_DIR, batch_size=1)

def denormalize(tensors, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalization parameters for pre-trained PyTorch models
     Denormalizes image tensors using mean and std """

    for c in range(3):
        tensors[:,c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors, min=0., max=1.)

model = prepare_model(num_classes=2)

model.train()
out = model(torch.randn((2, 3, 384, 384)))
print(out['out'].shape)


def step(model, epoch_num=None, loader=None, optimizer_fn=None, loss_fn=None, metric_fn=None, is_train=False, metric_name="iou"):

    loss_record   = MeanMetric()
    metric_record = MeanMetric()
    
    loader_len = len(loader)

    text = "Train" if is_train else "Valid"

    for data in tqdm(iterable=loader, total=loader_len, dynamic_ncols=True, desc=f"{text} :: Epoch: {epoch_num}"):
        
        if is_train:
            preds = model(data[0])["out"]
        else:
            with torch.no_grad():
                preds = model(data[0])["out"].detach()

        loss = loss_fn(preds, data[1])

        if is_train:
            optimizer_fn.zero_grad()
            loss.backward()
            optimizer_fn.step()

        metric = metric_fn(preds.detach(), data[1])

        loss_value = loss.detach().item()
        metric_value = metric.detach().item()
        
        loss_record.update(loss_value)
        metric_record.update(metric_value)

    current_loss   = loss_record.compute()
    current_metric = metric_record.compute()

    # print(f"\rEpoch {epoch:>03} :: TRAIN :: LOSS: {loss_record.compute()}, {metric_name.upper()}: {metric_record.compute()}\t\t\t\t", end="")

    return current_loss, current_metric



backbone_model_name = "mbv3" # mbv3 | r50 | r101

model = prepare_model(backbone_model=backbone_model_name, num_classes=NUM_CLASSES)
model.to(device)

# Dummy pass through the model
_ = model(torch.randn((2, 3, 384, 384), device=device))


train_loader, valid_loader = get_dataset(data_directory=DATA_DIR, batch_size=BATCH_SIZE)
train_loader = DeviceDataLoader(train_loader, device)
valid_loader = DeviceDataLoader(valid_loader, device)

metric_name = "iou"
use_dice = True if metric_name == "dice" else False 

metric_fn = Metric(num_classes=NUM_CLASSES, use_dice=use_dice).to(device)
loss_fn   = Loss(use_dice=use_dice).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



liveloss = PlotLosses()  

best_metric = 0.0

for epoch in range(1, NUM_EPOCHS + 1):

    logs = {}

    model.train()
    train_loss, train_metric = step(model, 
                                    epoch_num=epoch, 
                                    loader=train_loader, 
                                    optimizer_fn=optimizer, 
                                    loss_fn=loss_fn, 
                                    metric_fn=metric_fn, 
                                    is_train=True,
                                    metric_name=metric_name,
                                    )

    model.eval()
    valid_loss, valid_metric = step(model, 
                                    epoch_num=epoch, 
                                    loader=valid_loader, 
                                    loss_fn=loss_fn, 
                                    metric_fn=metric_fn, 
                                    is_train=False,
                                    metric_name=metric_name,
                                    )

    logs['loss']               = train_loss
    logs[metric_name]          = train_metric
    logs['val_loss']           = valid_loss
    logs[f'val_{metric_name}'] = valid_metric

    liveloss.update(logs)
    liveloss.send()

    if valid_metric >= best_metric:
        print("\nSaving model.....")
        torch.save(model.state_dict(), BESTMODEL_PATH)
        best_metric = valid_metric


class Trainer():
    def __init__(self, args):
        model = prepare_model(backbone_model=backbone_model_name, num_classes=NUM_CLASSES)
        model.to(device)

        # Dummy pass through the model
        _ = model(torch.randn((2, 3, 384, 384), device=device))


        train_loader, valid_loader = get_dataset(data_directory=DATA_DIR, batch_size=BATCH_SIZE)
        train_loader = DeviceDataLoader(train_loader, device)
        valid_loader = DeviceDataLoader(valid_loader, device)

        metric_name = "iou"
        use_dice = True if metric_name == "dice" else False 

        metric_fn = Metric(num_classes=NUM_CLASSES, use_dice=use_dice).to(device)
        loss_fn   = Loss(use_dice=use_dice).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



        liveloss = PlotLosses()  

        best_metric = 0.0


    def step(self, model, epoch_num=None, loader=None, optimizer_fn=None, loss_fn=None, metric_fn=None, is_train=False, metric_name="iou"):

        loss_record   = MeanMetric()
        metric_record = MeanMetric()
        
        loader_len = len(loader)

        text = "Train" if is_train else "Valid"

        for data in tqdm(iterable=loader, total=loader_len, dynamic_ncols=True, desc=f"{text} :: Epoch: {epoch_num}"):
            
            if is_train:
                preds = model(data[0])["out"]
            else:
                with torch.no_grad():
                    preds = model(data[0])["out"].detach()

            loss = loss_fn(preds, data[1])

            if is_train:
                optimizer_fn.zero_grad()
                loss.backward()
                optimizer_fn.step()

            metric = metric_fn(preds.detach(), data[1])

            loss_value = loss.detach().item()
            metric_value = metric.detach().item()
            
            loss_record.update(loss_value)
            metric_record.update(metric_value)

        current_loss   = loss_record.compute()
        current_metric = metric_record.compute()

        # print(f"\rEpoch {epoch:>03} :: TRAIN :: LOSS: {loss_record.compute()}, {metric_name.upper()}: {metric_record.compute()}\t\t\t\t", end="")

        return current_loss, current_metric