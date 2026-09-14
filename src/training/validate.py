"""Validation loop with mask-aware per-task metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

from src.evaluation.metrics import calculate_multitask_metrics


@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    *,
    task_names: Sequence[str] = ("cardiomegaly", "tuberculosis"),
    thresholds: float | Sequence[float] = 0.5,
    return_metrics: bool = False,
    max_batches: int | None = None,
):
    """Validate one epoch.

    By default this retains the old API and returns only a float loss. Set
    ``return_metrics=True`` to receive ``{"loss": ..., "metrics": ...}``.
    ``max_batches`` is intended only for pipeline smoke tests.
    """
    model.eval()
    device = torch.device(device)
    running_loss = 0.0
    processed_batches = 0
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []

    progress = tqdm(dataloader, desc="Validation", leave=False)
    for batch_index, (images, labels, masks) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()

        logits = model(images)
        loss = criterion(logits, labels, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError("Validation loss menjadi NaN atau inf.")

        running_loss += float(loss.item())
        processed_batches += 1
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
        all_masks.append(masks.cpu().numpy())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if processed_batches == 0:
        raise RuntimeError("Validation dataloader tidak menghasilkan batch.")

    epoch_loss = running_loss / processed_batches
    if not return_metrics:
        return epoch_loss

    metrics = calculate_multitask_metrics(
        labels=np.concatenate(all_labels, axis=0),
        probabilities=np.concatenate(all_probabilities, axis=0),
        masks=np.concatenate(all_masks, axis=0),
        task_names=task_names,
        thresholds=thresholds,
    )
    return {"loss": epoch_loss, "metrics": metrics}
