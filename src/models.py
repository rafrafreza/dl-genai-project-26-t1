import torch
import torch.nn as nn


# -----------------------
# CNN Model (Milestone-3)
# -----------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------
# CRNN Model with BiGRU (Milestone-4)
# -----------------------
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # BiGRU for sequence learning
        self.gru = nn.GRU(
            input_size=32 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layer
        self.fc = nn.Linear(128 * 2, 10)  # *2 because bidirectional

    def forward(self, x):
        x = self.cnn(x)

        b, c, h, w = x.size()
        x = x.view(b, w, c * h)

        x, _ = self.gru(x)
        x = x[:, -1, :]

        return self.fc(x)


# -----------------------
# Simple Transformer Model (Milestone-5)
# -----------------------
class AudioTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32 * 32,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(32 * 32, 10)

    def forward(self, x):
        x = self.cnn(x)

        b, c, h, w = x.size()
        x = x.view(b, w, c * h)

        x = self.transformer(x)
        x = x[:, -1, :]

        return self.fc(x)