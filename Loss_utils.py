# -- coding: utf-8 --
import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import min_max_normalize, downsample_upsample_labels
 
class DiceLoss(nn.Module):
    def __init__(self, num_classes, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def diceCoeff(self, pred, gt, eps=1e-5):
        r""" computational formula：
            dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
        """
        N = gt.size(0)
        pred = torch.sigmoid(pred).float()
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)

        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        loss =  (2 * intersection + eps) / (unionset + eps)
        loss = loss.clone().detach().requires_grad_(True)
        return loss.sum() / N

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(self.num_classes):
            class_dice.append(self.diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))
    
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice

def Compute_KDloss(Feas_teacher,Feas_student):
    Feas_teacher = [min_max_normalize(feats.pow(2).mean(1).unsqueeze(1)) for feats in Feas_teacher]
    Feas_student = [min_max_normalize(feats.pow(2).mean(1).unsqueeze(1)) for feats in Feas_student]
    
    KD_loss = sum([torch.mean((teacher - student).pow(2)) for teacher, student in zip(Feas_teacher, Feas_student)])

    return KD_loss

