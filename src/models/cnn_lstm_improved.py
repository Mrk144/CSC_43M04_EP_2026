# Rôle : Le cerveau avec Kaiming Init, GELU, Dropout et Mean Pooling
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models

class CNNLSTMImproved(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False, lstm_hidden_size: int = 512, dropout_p: float = 0.5):
        super().__init__()
        
        # 1. Backbone (pretrained=False pour Track A)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 2. Régularisation et Activation (GELU remplace ReLU)
        self.dropout = nn.Dropout(p=dropout_p)
        self.activation = nn.GELU() 

        # 3. LSTM
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)

        # 4. Classifieur avec Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(lstm_hidden_size, num_classes)
        )

        # 5. Application de l'initialisation (Kaiming / Orthogonal)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)
                elif 'bias' in name:
                    init.constant_(param.data, 0)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = video_batch.shape
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        # Extraction visuelle
        frame_features = self.backbone(frames)
        frame_features = self.activation(frame_features) # GELU
        frame_features = self.dropout(frame_features)    # Dropout
        
        # Séquence temporelle
        sequence = frame_features.view(batch_size, num_frames, -1)
        lstm_out, _ = self.lstm(sequence)

        # AMÉLIORATION : Mean Pooling temporel au lieu du dernier élément
        pooled_features = lstm_out.mean(dim=1)

        return self.classifier(pooled_features)