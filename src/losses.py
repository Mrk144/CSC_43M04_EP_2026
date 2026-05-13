import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithSmoothing(nn.Module):
    """
    Combinaison de Focal Loss et Label Smoothing.
    - Focal Loss : se concentre sur les exemples difficiles en réduisant le poids des faciles.
    - Label Smoothing : empêche le surapprentissage en lissant les cibles.
    """
    def __init__(self, smoothing=0.1, gamma=2.0):
        super(FocalLossWithSmoothing, self).__init__()
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits : prédictions brutes (B, num_classes)
        targets : indices des classes réelles (B,)
        """
        num_classes = logits.size(-1)
        
        # 1. Préparation du Label Smoothing (Soft Targets)
        # On ne calcule pas de gradients pour cette étape de préparation
        with torch.no_grad():
            # On distribue une petite probabilité (smoothing) sur toutes les classes
            smooth_target = torch.full_like(logits, self.smoothing / (num_classes - 1))
            # On assigne (1 - smoothing) à la classe correcte
            smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            
        # 2. Calcul des Log Probabilités via LogSoftmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 3. Cross Entropy avec les cibles lissées
        # CE = -target * log(prob)
        ce_loss = -smooth_target * log_probs
        
        # 4. Calcul du poids Focal
        # On repasse en probabilités réelles pour calculer le poids
        probs = torch.exp(log_probs)
        # Formule : (1 - p)^gamma
        # Plus p est proche de 1 (facile), plus le poids est proche de 0.
        focal_weight = (1 - probs) ** self.gamma
        
        # 5. Fusion et Moyenne
        # On applique le poids, on somme par échantillon, puis moyenne sur le batch
        loss = focal_weight * ce_loss
        return loss.sum(dim=-1).mean()