"""Save and restore reproducible training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model,
    save_path,
    *,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model state and optional training state in one checkpoint."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"Checkpoint tersimpan: {path}")


def load_checkpoint(
    model,
    checkpoint_path,
    device,
    *,
    optimizer=None,
    scheduler=None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load both new full checkpoints and older raw state-dict files."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics", {}),
            "extra": checkpoint.get("extra", {}),
        }
    else:
        model.load_state_dict(checkpoint, strict=strict)
        metadata = {"epoch": None, "metrics": {}, "extra": {}}

    print(f"Checkpoint dimuat: {checkpoint_path}")
    return metadata
