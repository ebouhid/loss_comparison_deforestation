import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # print(pred.shape)
        # print(target.shape)
        # Flatten the predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)

        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # Compute the focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)

        # Apply reduction
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss