import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithSmoothing(nn.Module):
    """Focal loss with label smoothing and optional per-class weights."""

    def __init__(
        self,
        smoothing: float = 0.1,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer(
                "class_weights", class_weights.float(), persistent=False
            )
        else:
            self.class_weights = None

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
        loss = focal_weight * ce_loss.sum(dim=-1).squeeze(-1)
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()
