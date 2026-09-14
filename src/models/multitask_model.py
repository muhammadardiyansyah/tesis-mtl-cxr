"""Shared-backbone multi-task model for chest X-ray classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.models.backbone import FeatureBackbone


DEFAULT_TASK_NAMES = ("cardiomegaly", "tuberculosis")


class MultiTaskModel(nn.Module):
    """One shared image encoder with one binary head for each task.

    ``forward`` returns raw logits with shape ``[batch_size, num_tasks]``.
    Sigmoid is intentionally not applied here because BCE-with-logits losses
    are numerically more stable. Apply ``torch.sigmoid`` only for inference.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
        task_names: Sequence[str] = DEFAULT_TASK_NAMES,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout harus berada pada rentang [0, 1).")

        normalized_names = tuple(str(name).strip().lower() for name in task_names)
        if not normalized_names or any(not name for name in normalized_names):
            raise ValueError("task_names tidak boleh kosong.")
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("Setiap nama task harus unik.")

        self.task_names = normalized_names
        self.backbone = FeatureBackbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )
        self.heads = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(self.backbone.output_dim, 1),
                )
                for task_name in self.task_names
            }
        )

    @property
    def num_tasks(self) -> int:
        return len(self.task_names)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(images)
        task_logits = [self.heads[name](features) for name in self.task_names]
        return torch.cat(task_logits, dim=1)

    def forward_dict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return one ``[batch]`` logit tensor per named task."""
        logits = self.forward(images)
        return {
            name: logits[:, index]
            for index, name in enumerate(self.task_names)
        }

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone.set_trainable(trainable)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone.get_gradcam_target_layer()

    def count_parameters(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)


# Backward-compatible alias in case an older notebook used this class name.
MultiTaskCXRModel = MultiTaskModel
