"""Training loop for the mask-aware multi-task CXR model."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.training.validate import validate_one_epoch
from src.utils.checkpoint import save_checkpoint


def _write_running_history(
    history: dict[str, list],
    history_path: Path | None,
    *,
    last_completed_epoch: int,
    status: str,
) -> None:
    """Persist progress atomically so interrupted cloud runs retain their log."""
    if history_path is None:
        return

    history_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = history_path.with_suffix(history_path.suffix + ".tmp")
    payload = {
        "status": status,
        "last_completed_epoch": last_completed_epoch,
        "history": history,
    }
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(history_path)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    *,
    scaler=None,
    amp_enabled: bool = False,
    grad_clip_norm: float | None = 1.0,
    max_batches: int | None = None,
) -> float:
    """Train for one epoch and return mean batch loss."""
    model.train()
    device = torch.device(device)
    amp_enabled = bool(amp_enabled and device.type == "cuda")
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
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
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels, masks)

        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss menjadi NaN atau inf.")

        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

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
    latest_checkpoint_path=None,
    history_path=None,
    scheduler=None,
    scaler=None,
    amp_enabled: bool = False,
    task_names=("cardiomegaly", "tuberculosis"),
    thresholds=0.5,
    grad_clip_norm: float | None = 1.0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    start_epoch: int = 1,
    history: dict[str, list] | None = None,
    best_val_loss: float = float("inf"),
    epochs_without_improvement: int = 0,
) -> dict[str, list]:
    """Train, validate, checkpoint every epoch, and support safe resuming."""
    if epochs < 1:
        raise ValueError("epochs minimal 1.")
    if patience < 1:
        raise ValueError("patience minimal 1.")
    if start_epoch < 1:
        raise ValueError("start_epoch minimal 1.")

    device = torch.device(device)
    model.to(device)
    amp_enabled = bool(amp_enabled and device.type == "cuda")
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if history is None:
        history = {}
    history.setdefault("train_loss", [])
    history.setdefault("val_loss", [])
    history.setdefault("val_metrics", [])
    history.setdefault("epoch_seconds", [])

    if start_epoch > epochs:
        print(f"Semua {epochs} epoch sudah selesai pada checkpoint.")
        return history

    stopped_early = False
    last_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch - 1)

        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
            max_batches=max_train_batches,
        )
        validation = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            task_names=task_names,
            thresholds=thresholds,
            return_metrics=True,
            max_batches=max_val_batches,
        )
        val_loss = float(validation["loss"])
        val_metrics = validation["metrics"]
        epoch_seconds = time.perf_counter() - epoch_started

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)
        history["epoch_seconds"].append(epoch_seconds)
        last_completed_epoch = epoch
        print(
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"Waktu: {epoch_seconds / 60:.1f} menit"
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
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

        checkpoint_extra: dict[str, Any] = {
            "task_names": list(task_names),
            "history": history,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "amp_enabled": amp_enabled,
        }
        checkpoint_metrics = {"val_loss": val_loss, "tasks": val_metrics}

        if improved and checkpoint_path is not None:
            save_checkpoint(
                model,
                Path(checkpoint_path),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=checkpoint_metrics,
                extra=checkpoint_extra,
            )

        if latest_checkpoint_path is not None:
            save_checkpoint(
                model,
                Path(latest_checkpoint_path),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=checkpoint_metrics,
                extra=checkpoint_extra,
            )

        _write_running_history(
            history,
            None if history_path is None else Path(history_path),
            last_completed_epoch=epoch,
            status="running",
        )

        if epochs_without_improvement >= patience:
            print("Early stopping dipicu.")
            stopped_early = True
            break

    _write_running_history(
        history,
        None if history_path is None else Path(history_path),
        last_completed_epoch=last_completed_epoch,
        status="early_stopped" if stopped_early else "completed",
    )
    return history
