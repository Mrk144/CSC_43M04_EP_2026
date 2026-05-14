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
        
        # Backbone (ResNet18)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.feature_projection = nn.Linear(feature_dim, d_model)
        
        # FIX 1: Learned Positional Embedding for exactly 4 frames
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # FIX 2: The classifier now expects (d_model * num_frames) because we flatten!
        self.classifier = nn.Linear(d_model * num_frames, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        batch_size, T, C, H, W = video_batch.shape
        
        x = video_batch.view(batch_size * T, C, H, W)
        features = self.backbone(x) 
        
        features = self.feature_projection(features) 
        features = features.view(batch_size, T, -1) 
        
        # Add learned position
        features = features + self.pos_embed
        features = self.transformer(features) 
        
        # FIX 3: Flatten instead of mean! Preserves strict temporal order.
        features = features.flatten(start_dim=1) # Shape: (B, T * d_model)
        
        return self.classifier(features)