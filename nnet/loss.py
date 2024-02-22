import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_2_onehot(matrix, num_classes=2):
    '''
    Perform one-hot encoding across the channel dimension.
    '''
    matrix = matrix.permute(0, 2, 3, 1)
    matrix = torch.argmax(matrix, dim=-1)
    matrix = torch.nn.functional.one_hot(matrix, num_classes=num_classes)
    matrix = matrix.permute(0, 3, 1, 2)

    return matrix


def intermediate_metric_calculation(predictions, targets, use_dice=False, smooth=1e-6, dims=(2, 3)):
    # dimscorresponding to image height and width: [B, C, H, W].
    
    # Intersection: |G ∩ P|. Shape: (batch_size, num_classes)
    intersection = (predictions * targets).sum(dim=dims) + smooth 

    # Summation: |G| + |P|. Shape: (batch_size, num_classes).
    summation = (predictions.sum(dim=dims) + targets.sum(dim=dims)) + smooth 
        
    if use_dice:
        # Dice Shape: (batch_size, num_classes) 
        metric = (2.0 * intersection) / summation
    else:
        # Union. Shape: (batch_size, num_classes)
        union = summation - intersection

        # IoU Shape: (batch_size, num_classes)
        metric = intersection /  union
        
    # Compute the mean over the remaining axes (batch and classes). 
    # Shape: Scalar
    total = metric.mean()
    
    return total

class Loss(nn.Module):
    def __init__(self, smooth=1e-6, use_dice=False):
        super().__init__()
        self.smooth = smooth
        self.use_dice = use_dice


    def forward(self, predictions, targets):
        # predictions --> (B, #C, H, W) unnormalized
        # targets     --> (B, #C, H, W) one-hot encoded

        # Normalize model predictions
        predictions = torch.sigmoid(predictions)

        # Calculate pixel-wise loss for both channels. Shape: Scalar
        pixel_loss = F.binary_cross_entropy(predictions, targets, reduction="mean")
        
        mask_loss  = 1 - intermediate_metric_calculation(predictions, targets, use_dice=self.use_dice, smooth=self.smooth)
        total_loss = mask_loss + pixel_loss
        
        return total_loss
    
class Metric(nn.Module):
    def __init__(self, args, smooth=1e-6, use_dice=False):
        super().__init__()
        self.num_classes = args.num_class
        self.smooth      = smooth
        self.use_dice    = use_dice
    
    def forward(self, predictions, targets):
        # predictions  --> (B, #C, H, W) unnormalized
        # targets      --> (B, #C, H, W) one-hot encoded 

        # Converting unnormalized predictions into one-hot encoded across channels.
        # Shape: (B, #C, H, W) 
        predictions = convert_2_onehot(predictions, num_classes=self.num_classes) # one hot encoded

        metric = intermediate_metric_calculation(predictions, targets, use_dice=self.use_dice, smooth=self.smooth)
        
        # Compute the mean over the remaining axes (batch and classes). Shape: Scalar
        return metric
    
    