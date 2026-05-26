import torch.nn as nn

from torchvision import models


class SingleTaskModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet18(
            weights="DEFAULT"
        )

        in_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Linear(
            in_features,
            1
        )

    def forward(self, x):

        return self.backbone(x)