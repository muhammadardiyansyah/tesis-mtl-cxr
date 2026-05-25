import torch
import torch.nn as nn


class MaskedBCELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(
            reduction="none"
        )

    def forward(
        self,
        outputs,
        targets,
        mask
    ):

        loss = self.bce(
            outputs,
            targets
        )

        loss = loss * mask

        loss = loss.sum() / mask.sum()

        return loss