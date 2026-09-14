"""Training loop for the mask-aware multi-task CXR model."""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from src.training.validate import validate_one_epoch
from src.utils.checkpoint import save_checkpoint


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    *,
    grad_clip_norm: float | None = 1.0,
    max_batches: int | None = None,
) -> float:
    """Train for one epoch and return mean batch loss.

    ``max_batches`` must remain ``None`` for a real experiment; it is provided
    only for quick end-to-end smoke tests.
    """
    model.train()
    device = torch.device(device)
    running_loss = 0.0
    processed_batches = 0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch_index, (images, labels, masks) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss menjadi NaN atau inf.")

        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        running_loss += float(loss.item())
        processed_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if processed_batches == 0:
        raise RuntimeError("Training dataloader tidak menghasilkan batch.")
    return running_loss / processed_batches


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs: int = 10,
    patience: int = 3,
    checkpoint_path=None,
    *,
    scheduler=None,
    task_names=("cardiomegaly", "tuberculosis"),
    thresholds=0.5,
    grad_clip_norm: float | None = 1.0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, list]:
    """Train, validate, early-stop, and retain the best validation checkpoint."""
    if epochs < 1:
        raise ValueError("epochs minimal 1.")
    if patience < 1:
        raise ValueError("patience minimal 1.")

    model.to(device)
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": [],
    }
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
            max_batches=max_train_batches,
        )
        validation = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            task_names=task_names,
            thresholds=thresholds,
            return_metrics=True,
            max_batches=max_val_batches,
        )
        val_loss = float(validation["loss"])
        val_metrics = validation["metrics"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                save_checkpoint(
                    model,
                    Path(checkpoint_path),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics={"val_loss": val_loss, "tasks": val_metrics},
                    extra={"task_names": list(task_names)},
                )
        else:
            epochs_without_improvement += 1
            print(
                "Early stopping: "
                f"{epochs_without_improvement}/{patience} epoch tanpa perbaikan"
            )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if epochs_without_improvement >= patience:
            print("Early stopping dipicu.")
            break

    return history
