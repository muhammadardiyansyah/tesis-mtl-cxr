import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCELoss(nn.Module):

    def __init__(
        self,
        pos_weight=None
    ):

        super().__init__()

        self.pos_weight = pos_weight

    def forward(
        self,
        outputs,
        targets,
        mask
    ):

        loss = F.binary_cross_entropy_with_logits(
            outputs,
            targets,
            reduction="none",
            pos_weight=self.pos_weight
        )

        loss = loss * mask

        if mask.sum() == 0:

            return torch.tensor(
                0.0,
                device=outputs.device,
                requires_grad=True
            )

        return loss.sum() / mask.sum()