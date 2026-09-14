from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCELoss(nn.Module):
    """
    Binary cross-entropy untuk multi-task learning
    dengan label parsial.

    Setiap task dihitung secara terpisah agar task
    dengan jumlah data terbesar tidak otomatis
    mendominasi total loss.
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        task_weight: Optional[Sequence[float]] = None,
    ):
        super().__init__()

        if pos_weight is None:
            pos_weight = torch.ones(
                2,
                dtype=torch.float32,
            )
        else:
            pos_weight = torch.as_tensor(
                pos_weight,
                dtype=torch.float32,
            )

        if task_weight is None:
            task_weight_tensor = torch.ones(
                2,
                dtype=torch.float32,
            )
        else:
            task_weight_tensor = torch.as_tensor(
                task_weight,
                dtype=torch.float32,
            )

        if pos_weight.numel() != 2:
            raise ValueError(
                "pos_weight harus mempunyai dua nilai: "
                "[kardiomegali, tuberkulosis]"
            )

        if task_weight_tensor.numel() != 2:
            raise ValueError(
                "task_weight harus mempunyai dua nilai: "
                "[kardiomegali, tuberkulosis]"
            )

        self.register_buffer(
            "pos_weight",
            pos_weight,
        )

        self.register_buffer(
            "task_weight",
            task_weight_tensor,
        )

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if outputs.shape != targets.shape:
            raise ValueError(
                "Ukuran outputs dan targets berbeda: "
                f"{outputs.shape} vs {targets.shape}"
            )

        if mask.shape != targets.shape:
            raise ValueError(
                "Ukuran mask dan targets berbeda: "
                f"{mask.shape} vs {targets.shape}"
            )

        element_loss = (
            F.binary_cross_entropy_with_logits(
                outputs,
                targets,
                reduction="none",
                pos_weight=self.pos_weight,
            )
        )

        mask = mask.to(
            dtype=element_loss.dtype
        )

        masked_loss = element_loss * mask

        valid_count_per_task = mask.sum(
            dim=0
        )

        active_tasks = (
            valid_count_per_task > 0
        ).to(element_loss.dtype)

        loss_per_task = (
            masked_loss.sum(dim=0)
            / valid_count_per_task.clamp_min(1.0)
        )

        active_task_weights = (
            self.task_weight
            * active_tasks
        )

        total_active_weight = (
            active_task_weights.sum()
        )

        if total_active_weight.item() == 0:
            # Menghasilkan zero loss yang tetap terhubung
            # dengan computation graph.
            return outputs.sum() * 0.0

        total_loss = (
            loss_per_task
            * active_task_weights
        ).sum() / total_active_weight

        return total_loss