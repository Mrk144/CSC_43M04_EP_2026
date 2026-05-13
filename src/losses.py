# Rôle : Contient la nouvelle fonction de perte (Focal Loss + Label Smoothing)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, gamma=2.0):
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        with torch.no_grad():
            smooth_target = torch.full_like(logits, self.smoothing / (num_classes - 1))
            smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -smooth_target * log_probs
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.sum(dim=-1).mean()