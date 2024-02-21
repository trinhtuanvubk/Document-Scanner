import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from torchvision.utils import make_grid, save_image
from torchmetrics import MeanMetric
from livelossplot import PlotLosses

from dataloader.dataloader import get_dataset, DeviceDataLoader
from nnet.model import prepare_model
from nnet.loss import Loss, Metric
from utils.args import get_args

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = prepare_model(backbone_model=args.backbone_model, num_classes=args.num_class)
        self.model.to(args.device)

        train_loader, valid_loader = get_dataset(data_directory=DATA_DIR, batch_size=BATCH_SIZE)
        self.train_loader = DeviceDataLoader(train_loader, args.device)
        self.valid_loader = DeviceDataLoader(valid_loader, args.device)

        # metric_name = "iou"
        use_dice = True if args.metric_name == "dice" else False 
        self.metric_fn = Metric(num_classes=NUM_CLASSES, use_dice=use_dice).to(args.device)
        self.loss_fn = Loss(use_dice=use_dice).to(args.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.liveloss = PlotLosses()  

        self.best_metric = 0.0
        self.metric_name = args.metric_name
        self.checkpoint_path = args.checkpoint_path


    def step(self, epoch_num=None, loader=None, is_train=False):

        loss_record   = MeanMetric()
        metric_record = MeanMetric()
        
        loader_len = len(loader)

        text = "Train" if is_train else "Valid"

        for data in tqdm(iterable=loader, total=loader_len, dynamic_ncols=True, desc=f"{text} :: Epoch: {epoch_num}"):
            
            if is_train:
                preds = self.model(data[0])["out"]
            else:
                with torch.no_grad():
                    preds = self.model(data[0])["out"].detach()

            loss = self.loss_fn(preds, data[1])

            if is_train:
                self.optimizer_fn.zero_grad()
                self.loss.backward()
                self.optimizer_fn.step()

            metric = self.metric_fn(preds.detach(), data[1])

            loss_value = loss.detach().item()
            metric_value = metric.detach().item()
            
            loss_record.update(loss_value)
            metric_record.update(metric_value)

        current_loss   = loss_record.compute()
        current_metric = metric_record.compute()

        # print(f"\rEpoch {epoch:>03} :: TRAIN :: LOSS: {loss_record.compute()}, {metric_name.upper()}: {metric_record.compute()}\t\t\t\t", end="")

        return current_loss, current_metric

    def fit(self):
        for epoch in range(1, self.args.num_epoch + 1):

            logs = {}
            self.model.train()
            train_loss, train_metric = self.step(self.model, 
                                            epoch_num=epoch, 
                                            loader=self.train_loader, 
                                            is_train=True,
                                            )

            self.model.eval()
            valid_loss, valid_metric = self.step(self.model, 
                                            epoch_num=epoch, 
                                            loader=self.valid_loader, 
                                            is_train=False,
                                            )

            logs['loss']               = train_loss
            logs[metric_name]          = train_metric
            logs['val_loss']           = valid_loss
            logs[f'val_{metric_name}'] = valid_metric

            self.liveloss.update(logs)
            self.liveloss.send()

            if valid_metric >= best_metric:
                print("\nSaving model.....")
                torch.save(model.state_dict(), BESTMODEL_PATH)
                self.best_metric = valid_metric


def train(args):
    seed_everything()
    trainer = Trainer(args)
    trainer.fit()
