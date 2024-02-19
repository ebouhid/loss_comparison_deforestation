import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
        super(BinaryTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten the tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate true positives, false positives, and false negatives
        tp = torch.sum(pred_flat * target_flat)
        fp = torch.sum((1 - target_flat) * pred_flat)
        fn = torch.sum(target_flat * (1 - pred_flat))

        # Calculate Tversky coefficient
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Calculate Tversky loss as 1 - Tversky coefficient
        tversky_loss = 1 - tversky

        return tversky_loss

# # Example usage:
# # Assuming raw_model_output is the raw output from your model without the final activation
# # and target is the ground truth binary mask
# raw_model_output = torch.randn(B, 1, H, W, requires_grad=True)
# target = torch.randint(0, 2, (B, 1, H, W))

# tversky_loss_criterion = BinaryTverskyLoss(alpha=0.3, beta=0.7)
# loss = tversky_loss_criterion(torch.sigmoid(raw_model_output), target)
# loss.backward()
