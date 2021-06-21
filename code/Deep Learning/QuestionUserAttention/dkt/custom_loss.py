
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        loss = nn.BCELoss(reduction="none")
        bce_loss = loss(pred, target)
        auc = roc_auc_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())
        auc_loss = 1 - auc
        
        return bce_loss + auc_loss