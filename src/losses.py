import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, gamma=2.0):
        super(FocalLossWithSmoothing, self).__init__()
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        
        # 1. Calculate probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # 2. Get the probability of the ACTUAL ground truth class (pt)
        # Shape: (B, 1)
        pt = probs.gather(1, targets.unsqueeze(1))
        
        # 3. Calculate the Focal Weight based ONLY on the true class
        # Shape: (B, 1)
        focal_weight = (1 - pt) ** self.gamma
        
        # 4. Prepare Label Smoothing (Soft Targets)
        with torch.no_grad():
            smooth_target = torch.full_like(logits, self.smoothing / (num_classes - 1))
            smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            
        # 5. Calculate Smoothed Cross Entropy
        # Shape: (B, C)
        ce_loss = -smooth_target * log_probs
        
        # 6. Apply focal weight to the total CE loss for each sample, then average
        # We sum over the classes (dim=-1) to get total CE per sample, then apply the weight
        loss = focal_weight * ce_loss.sum(dim=-1, keepdim=True)
        return loss.mean()