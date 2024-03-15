import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', debug=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.debug = debug

    def forward(self, pred, target):
        # print(pred.shape)
        # print(target.shape)
        # Flatten the predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        pt = torch.where(target == 1, pred, 1 - pred)

        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        # Compute the cross-entropy loss
        assert pred.shape == target.shape, f'pred shape: {pred.shape}, target shape: {target.shape} mismatch!'
        assert pt.shape == target.shape, f'pt shape: {pt.shape}, target shape: {target.shape} mismatch!'
        assert torch.all(pred >= 0), f'pred has negative values: {pred}'
        assert torch.all(pred <= 1), f'pred has values greater than 1: {pred}'
        assert torch.all(pt >= 0), f'pt has negative values: {pt}'
        assert torch.all(pt <= 1), f'pt has values greater than 1: {pt}'
        assert torch.all(target >= 0), f'target has negative values: {target}'
        assert torch.all(target <= 1), f'target has values greater than 1: {target}'
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')

        if self.debug:
            print(f'pred range: {pred.min()} - {pred.max()}')
            print(f'pt range: {pt.min()} - {pt.max()}')
            print(f'target range: {target.min()} - {target.max()}')
            print(f'pt shape: {pt.shape}')
            print(f'target shape: {target.shape}')
            print(f'ce_loss shape: {ce_loss.shape}')
            print(f'CE Loss: {ce_loss}')
            print(f'alpha_t shape: {alpha_t.shape}')
            print(f'alpha_t range: {alpha_t.min()} - {alpha_t.max()}')

        # Compute the focal loss
        focal_loss = (alpha_t * ((1 - pt) ** self.gamma) * ce_loss)

        if self.debug:
            print(f'Focal Loss range: {focal_loss.min()} - {focal_loss.max()}')

        # Apply reduction
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        if self.debug:
            print(f'Focal Loss: {focal_loss}')

        return focal_loss