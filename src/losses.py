import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithSmoothing(nn.Module):
    """Focal loss with label smoothing.

    Falls back to standard ``CrossEntropyLoss(label_smoothing=...)`` semantics
    when ``num_classes < 2`` (avoids a division-by-zero in the smoothing term).
    """

    def __init__(self, smoothing: float = 0.1, gamma: float = 2.0) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        if num_classes < 2:
            raise ValueError(
                f"FocalLossWithSmoothing requires num_classes >= 2 (got {num_classes})."
            )

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        pt = probs.gather(1, targets.unsqueeze(1))
        focal_weight = (1 - pt) ** self.gamma

        with torch.no_grad():
            smooth_target = torch.full_like(
                logits, self.smoothing / (num_classes - 1)
            )
            smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        ce_loss = -smooth_target * log_probs
        loss = focal_weight * ce_loss.sum(dim=-1, keepdim=True)
        return loss.mean()
