import torch
import torch.nn.functional as F
import torch.nn as nn


def get_criterion(pred, target):
    loss = nn.BCELoss(reduction="none")
    return loss(pred, target)

def epoch_update_gamma(y_true, y_pred, epoch=-1, delta=2):
    """
    Calculate gamma from last epoch's targets and predictions.
    Gamma is updated at the end of each epoch.
    y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
    y_pred: `Tensor` . Predictions.
    """
    pos = y_pred[y_true==1]
    neg = y_pred[y_true==0] # yo pytorch, no boolean tensors or operators?  Wassap?
    
    # subsample the training set for performance
    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]
    
    pos_expand = pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
    neg_expand = neg.repeat(ln_pos)
    
    diff = neg_expand - pos_expand
    
    Lp = diff[diff>0] # because we're taking positive diffs, we got pos and neg flipped.
    ln_Lp = Lp.shape[0] - 1
    
    diff_neg = -1.0 * diff[diff<0]
    diff_neg = diff_neg.sort()[0]
    
    ln_neg = diff_neg.shape[0]-1
    ln_neg = max([ln_neg, 0])
    
    left_wing = int(ln_Lp*delta)
    left_wing = max([0, left_wing])
    left_wing = min([ln_neg, left_wing])
    
    default_gamma = torch.tensor(0.2, dtype=torch.float).cuda()
    if diff_neg.shape[0] > 0:
        gamma = diff_neg[left_wing]
    else:
        gamma = default_gamma # default=torch.tensor(0.2, dtype=torch.float).cuda() #zoink
    
    return gamma
    

def roc_star_loss(_y_true, y_pred, gamma, _epoch_true, epoch_pred):
    """
    Nearly direct loss function for AUC.
    See article,
    C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
    https://github.com/iridiumblue/articles/blob/master/roc_star.md
        _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        gamma  : `Float` Gamma, as derived from last epoch.
        _epoch_true: `Tensor`.  Targets (labels) from last epoch.
        epoch_pred : `Tensor`.  Predicions from last epoch.
    """
    #convert labels to boolean
    y_true = (_y_true>=0.50)
    epoch_true = (_epoch_true>=0.50)

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true)== 0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred)*1e-8

    pos = y_pred[y_true]
    neg = y_pred[~y_true]

    epoch_pos = epoch_pred[epoch_true]
    epoch_neg = epoch_pred[~epoch_true]
    
    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    cap_pos = epoch_pos.shape[0]
    cap_neg = epoch_neg.shape[0]
    
    # sum positive batch elements agaionst (subsampled) negative elements
    if ln_pos > 0:
        pos_expand = pos.view(-1,1).expand(-1, cap_neg).reshape(-1)
        neg_expand = epoch_neg.repeat(ln_pos)

        diff2 = neg_expand - pos_expand + gamma
        l2 = diff2[diff2>0]
        m2 = l2 * l2
        len2 = l2.shape[0]
    else:
        m2 = torch.tensor([0], dtype=torch.float).cuda()
        len2 = 0
        
    # Similarly, compare negative batch elements against (subsampled) positive elements
    if ln_neg > 0:
        pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
        neg_expand = neg.repeat(cap_pos)

        diff3 = neg_expand - pos_expand + gamma
        l3 = diff3[diff3>0]
        m3 = l3*l3
        len3 = l3.shape[0]
    else:
        m3 = torch.tensor([0], dtype=torch.float).cuda()
        len3=0

    if (torch.sum(m2) + torch.sum(m3)) != 0:
        res2 = torch.sum(m2)/(len2+1e-8) + torch.sum(m3)/(len3+1e-8)
    else:
        res2 = torch.sum(m2)/(len2+1e-8) + torch.sum(m3)/(len3+1e-8)

    res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

    return res2
