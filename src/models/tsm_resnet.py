import torch
import torch.nn as nn
from torchvision import models

def temporal_shift(x: torch.Tensor, num_frames: int, fold_div: int = 8) -> torch.Tensor:
    """
    Applique le Temporal Shift Module (TSM) sur un tenseur de features.
    
    Args:
        x: Tenseur de features de forme (Batch * Temps, Canaux, Hauteur, Largeur).
        num_frames: Le nombre de frames par vidéo (T).
        fold_div: La proportion des canaux à décaler (1/fold_div). 8 est le standard.
    """
    nt, c, h, w = x.size()
    batch_size = nt // num_frames
    
    # On reformate pour séparer le Batch et le Temps : (B, T, C, H, W)
    x = x.view(batch_size, num_frames, c, h, w)

    # On crée un tenseur vide pour stocker le résultat
    out = torch.zeros_like(x)
    
    # On calcule combien de canaux représentent 1/8 du total
    fold = c // fold_div

    # 1. Décalage vers le passé (les canaux [0 : fold] reculent d'un pas)
    out[:, :-1, :fold] = x[:, 1:, :fold]
    
    # 2. Décalage vers le futur (les canaux [fold : 2*fold] avancent d'un pas)
    out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
    
    # 3. Le reste (les canaux [2*fold : fin]) ne bouge pas
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

    # On remet le tenseur dans la forme (B*T, C, H, W) pour les convolutions 2D
    return out.view(nt, c, h, w)


class TSMResNet(nn.Module):
    """
    Un ResNet18 classique équipé du Temporal Shift Module.
    """
    def __init__(self, num_classes: int, num_frames: int = 4, pretrained: bool = False):
        super().__init__()
        self.num_frames = num_frames
        
        # 1. On charge le backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # 2. On découpe le ResNet pour injecter le TSM entre les blocs principaux
        # Le premier bloc contient la convolution initiale et le maxpool
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 3. Couches de classification
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # On ajoute un peu de dropout pour régulariser
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

        # Initialisation si entraînement from scratch
        if not pretrained:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        # video_batch shape: (Batch, Temps, Canaux, Hauteur, Largeur)
        batch_size, num_frames, channels, height, width = video_batch.shape
        
        # On fusionne B et T pour traiter chaque image comme indépendante
        x = video_batch.reshape(batch_size * num_frames, channels, height, width)

        # Passage dans les blocs avec TSM intercalé
        x = self.layer1(x)
        x = temporal_shift(x, self.num_frames) # L'information temporelle circule ici
        
        x = self.layer2(x)
        x = temporal_shift(x, self.num_frames) # Et ici
        
        x = self.layer3(x)
        x = temporal_shift(x, self.num_frames) # Et ici
        
        x = self.layer4(x)
        
        # Pooling spatial : on réduit chaque image à un vecteur
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)

        # On sépare à nouveau B et T : (Batch, Temps, Features)
        x = x.view(batch_size, num_frames, -1)
        
        # Pooling Temporel : on fait la moyenne sur le temps (Consensus)
        x = x.mean(dim=1)
        
        # Classification finale
        logits = self.fc(x)
        return logits