import torch
import torch.nn as nn

from torchvision import models


class BaselineModel(nn.Module):

    def __init__(self):

        super(BaselineModel, self).__init__()

        self.backbone = models.resnet18(
            weights="IMAGENET1K_V1"
        )

        in_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Linear(
            in_features,
            2
        )

    def forward(self, x):

        outputs = self.backbone(x)

        return outputs