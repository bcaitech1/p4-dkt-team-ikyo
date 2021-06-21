import torch.nn as nn
from .custom_loss import CustomLoss


def get_criterion(pred, target):
    loss = nn.BCELoss(reduction="none")
    return loss(pred, target[:,9:])