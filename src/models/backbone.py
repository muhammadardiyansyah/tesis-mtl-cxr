"""Shared feature backbones for the multi-task chest X-ray model.

The classifier supplied by torchvision is removed so this module returns one
feature vector per image. Classification heads are defined separately in
``multitask_model.py``.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torchvision import models


SUPPORTED_BACKBONES: Final[tuple[str, ...]] = (
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "vgg16",
    "vgg19",
    "vit_b_16",
)


class FeatureBackbone(nn.Module):
    """Torchvision backbone that outputs a two-dimensional feature tensor.

    Parameters
    ----------
    name:
        Backbone architecture name. See ``SUPPORTED_BACKBONES``.
    pretrained:
        Load ImageNet-1K weights when ``True``. The first use may download the
        weights, so use ``False`` for an offline smoke test.
    freeze:
        Freeze every backbone parameter at construction time when ``True``.
    """

    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.name = name.lower().strip()
        self.model, self.output_dim = self._build(self.name, pretrained)
        self.set_trainable(not freeze)

    @staticmethod
    def _build(name: str, pretrained: bool) -> tuple[nn.Module, int]:
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            output_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            output_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
            output_dim = model.classifier.in_features
            model.classifier = nn.Identity()
        elif name == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            model = models.efficientnet_b0(weights=weights)
            output_dim = model.classifier[-1].in_features
            model.classifier = nn.Identity()
        elif name == "vgg16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            model = models.vgg16(weights=weights)
            output_dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
        elif name == "vgg19":
            weights = models.VGG19_Weights.DEFAULT if pretrained else None
            model = models.vgg19(weights=weights)
            output_dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
        elif name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            model = models.vit_b_16(weights=weights)
            output_dim = model.heads.head.in_features
            model.heads = nn.Identity()
        else:
            choices = ", ".join(SUPPORTED_BACKBONES)
            raise ValueError(f"Backbone '{name}' tidak didukung. Pilihan: {choices}")

        return model, int(output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features

    def set_trainable(self, trainable: bool) -> None:
        """Enable or disable gradient calculation for all backbone parameters."""
        for parameter in self.model.parameters():
            parameter.requires_grad = trainable

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return a suitable final convolutional layer for Grad-CAM."""
        if self.name.startswith("resnet"):
            return self.model.layer4[-1]
        if self.name == "densenet121":
            return self.model.features.norm5
        if self.name == "efficientnet_b0":
            return self.model.features[-1]
        if self.name.startswith("vgg"):
            conv_layers = [
                layer for layer in self.model.features if isinstance(layer, nn.Conv2d)
            ]
            return conv_layers[-1]
        raise NotImplementedError(
            "Grad-CAM untuk vit_b_16 memerlukan reshape_transform khusus."
        )
