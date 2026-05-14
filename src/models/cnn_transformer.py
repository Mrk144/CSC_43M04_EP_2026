import torch
import torch.nn as nn
from torchvision import models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Temps, Features)
        return x + self.pe[:, :x.size(1)]

class CNNTransformer(nn.Module):
    def __init__(self, num_classes: int, num_frames: int = 4, pretrained: bool = False, d_model: int = 512, nhead: int = 8):
        super().__init__()
        
        # 1. Backbone (ResNet18)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 2. Projection pour correspondre à la dimension du Transformer
        self.feature_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_frames)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=1024, 
            dropout=0.3, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Classifieur
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        batch_size, T, C, H, W = video_batch.shape
        
        # Extraction des features spatiales
        x = video_batch.view(batch_size * T, C, H, W)
        features = self.backbone(x) # (B*T, 512)
        
        # Préparation pour le Transformer
        features = self.feature_projection(features) # (B*T, d_model)
        features = features.view(batch_size, T, -1) # (B, T, d_model)
        
        # Ajout du temps (Position) et passage dans le Transformer
        features = self.pos_encoder(features)
        features = self.transformer(features) # (B, T, d_model)
        
        # Pooling temporel (Moyenne sur les frames)
        features = features.mean(dim=1)
        
        return self.classifier(features)