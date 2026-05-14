import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models

class CNNLSTMImproved(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False, lstm_hidden_size: int = 512, dropout_p: float = 0.5):
        super().__init__()
        
        # --- BASELINE (Conservé) ---
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # --- NOUVEAU (TRACK A) : Régularisation et Activation ---
        self.dropout = nn.Dropout(p=dropout_p)           # Empêche d'apprendre par coeur
        self.activation = nn.GELU()                     # Activation plus performante

        # --- BASELINE (Conservé) ---
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)

        # --- NOUVEAU (TRACK A) : Classifieur robuste ---
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),                    # Dropout de sortie
            nn.Linear(lstm_hidden_size, num_classes)
        )

        # --- NOUVEAU (TRACK A) : Initialisation experte ---
        self.apply(self._initialize_weights)            # Kaiming / Orthogonal

    # ==========================================================================
    # === NOUVELLE FONCTION : Initialisation des poids ===
    # ==========================================================================
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data) # Crucial pour la mémoire longue
                elif 'bias' in name:
                    init.constant_(param.data, 0)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = video_batch.shape
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        # --- BASELINE + NOUVEAU (Activation/Dropout) ---
        frame_features = self.backbone(frames)
        frame_features = self.activation(frame_features)
        frame_features = self.dropout(frame_features)
        
        sequence = frame_features.view(batch_size, num_frames, -1)
        lstm_out, _ = self.lstm(sequence)

        last_hidden = lstm_out[:, -1, :]

        return self.classifier(last_hidden)