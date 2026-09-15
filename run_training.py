"""Command-line entry point for reproducible MTL CXR experiments."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.datasets.dataloaders import build_dataloaders
from src.losses.masked_loss import MaskedBCELoss
from src.models.multitask_model import MultiTaskModel
from src.training.train import train_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import get_config_path, load_config
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the mask-aware multi-task CXR model."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path config utama, relatif terhadap root proyek.",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "full"),
        default=None,
        help=(
            "smoke untuk uji singkat; pilot untuk satu epoch lengkap; "
            "full untuk eksperimen sebenarnya."
        ),
    )
    parser.add_argument(
        "--allow-cpu-full",
        action="store_true",
        help="Izinkan full training pada CPU (biasanya sangat lambat).",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Lanjutkan pilot/full dari checkpoint latest. Tanpa nilai memakai "
            "checkpoint latest default; path eksplisit juga dapat diberikan."
        ),
    )
    parser.add_argument(
        "--no-local-config",
        action="store_true",
        help="Abaikan configs/config.local.yaml, berguna pada Kaggle/cloud.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower().strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("training.device harus berisi auto, cpu, atau cuda.")
    return torch.device(requested)


def build_optimizer(model, config: dict):
    optimizer_config = config.get("optimizer", {})
    training_config = config["training"]
    name = str(optimizer_config.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError("Saat ini optimizer yang didukung adalah adamw.")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )


def build_scheduler(optimizer, config: dict):
    scheduler_config = config.get("scheduler", {})
    name = str(scheduler_config.get("name", "none")).lower()
    if name in {"none", "null", ""}:
        return None
    if name != "reduce_on_plateau":
        raise ValueError("Scheduler yang didukung: none atau reduce_on_plateau.")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config.get("factor", 0.5)),
        patience=int(scheduler_config.get("patience", 2)),
        min_lr=float(scheduler_config.get("min_learning_rate", 1e-6)),
    )


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        return None
    return value


def print_task_metrics(metrics: dict) -> None:
    print("\nMetrik validation terakhir:")
    for task_name, values in metrics.items():
        print(
            f"- {task_name}: n={values['n']}, "
            f"AUC={values['roc_auc']:.4f}, F1={values['f1']:.4f}, "
            f"sensitivitas={values['recall']:.4f}, "
            f"spesifisitas={values['specificity']:.4f}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config, use_local_config=not args.no_local_config)
    training_config = config["training"]
    model_config = config["model"]
    multitask_config = config["multitask"]
    experiment_config = config["experiment"]

    mode = args.mode or str(experiment_config.get("default_mode", "smoke"))
    device = resolve_device(training_config.get("device", "auto"))
    if mode == "full" and device.type == "cpu" and not args.allow_cpu_full:
        raise RuntimeError(
            "Full training diblokir karena hanya CPU yang tersedia. "
            "Gunakan mesin GPU, atau tambahkan --allow-cpu-full jika benar-benar sadar risikonya."
        )

    seed = int(training_config.get("random_seed", 42))
    set_global_seed(seed, bool(training_config.get("deterministic", True)))
    task_names = tuple(multitask_config["task_names"])
    pos_weights = [
        float(config["class_balance"][task_name]["pos_weight"])
        for task_name in task_names
    ]
    task_weights = [
        float(multitask_config["task_weights"][task_name])
        for task_name in task_names
    ]

    smoke_config = experiment_config.get("smoke", {})
    pilot_config = experiment_config.get("pilot", {})
    if mode == "smoke":
        epochs = int(smoke_config.get("epochs", 1))
        max_train_batches = int(smoke_config.get("train_batches", 2))
        max_val_batches = int(smoke_config.get("val_batches", 2))
        pretrained = bool(smoke_config.get("use_pretrained_weights", False))
        checkpoint_path = None
        latest_checkpoint_path = None
        history_filename = None
    elif mode == "pilot":
        epochs = int(pilot_config.get("epochs", 1))
        max_train_batches = None
        max_val_batches = None
        pretrained = bool(pilot_config.get("use_pretrained_weights", True))
        checkpoint_path = (
            get_config_path(config, "checkpoints")
            / pilot_config.get(
                "checkpoint_filename",
                "resnet18_mtl_pilot_v1_best.pt",
            )
        )
        latest_checkpoint_path = (
            get_config_path(config, "checkpoints")
            / pilot_config.get(
                "latest_checkpoint_filename",
                "resnet18_mtl_pilot_v1_latest.pt",
            )
        )
        history_filename = pilot_config.get(
            "history_filename",
            "resnet18_mtl_pilot_v1_history.json",
        )
    else:
        epochs = int(training_config["epochs"])
        max_train_batches = None
        max_val_batches = None
        pretrained = bool(model_config.get("pretrained", True))
        checkpoint_path = (
            get_config_path(config, "checkpoints")
            / experiment_config["checkpoint_filename"]
        )
        latest_checkpoint_path = (
            get_config_path(config, "checkpoints")
            / experiment_config.get(
                "latest_checkpoint_filename",
                "resnet18_mtl_baseline_v1_latest.pt",
            )
        )
        history_filename = experiment_config["history_filename"]

    history_path = None
    if history_filename is not None:
        history_path = get_config_path(config, "outputs") / history_filename

    amp_enabled = bool(
        device.type == "cuda" and training_config.get("mixed_precision", True)
    )

    print("=" * 68)
    print("MTL CXR TRAINING")
    print("=" * 68)
    print(f"Mode       : {mode}")
    print(f"Device     : {device}")
    print(f"Backbone   : {model_config['backbone']}")
    print(f"Pretrained : {pretrained}")
    print(f"Epoch      : {epochs}")
    print(f"Task       : {', '.join(task_names)}")
    print(f"AMP        : {amp_enabled}")
    if device.type == "cuda":
        print(f"GPU aktif  : {torch.cuda.get_device_name(0)}")
        print(f"GPU tersedia: {torch.cuda.device_count()} (pelatihan memakai 1 GPU)")

    loaders = build_dataloaders(config)
    model = MultiTaskModel(
        backbone_name=model_config["backbone"],
        pretrained=pretrained,
        dropout=float(model_config.get("dropout", 0.2)),
        task_names=task_names,
        freeze_backbone=bool(model_config.get("freeze_backbone", False)),
    ).to(device)
    criterion = MaskedBCELoss(
        pos_weight=torch.tensor(pos_weights, device=device),
        task_weight=task_weights,
    ).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    resume_history = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    resumed_from = None
    if args.resume is not None:
        if mode == "smoke":
            raise ValueError("--resume hanya didukung untuk mode pilot atau full.")
        if args.resume == "auto":
            resume_path = latest_checkpoint_path
        else:
            resume_path = Path(args.resume).expanduser()
            if not resume_path.is_absolute():
                resume_path = Path(__file__).resolve().parent / resume_path
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint resume tidak ditemukan: {resume_path}")

        resume_metadata = load_checkpoint(
            model,
            resume_path,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        completed_epoch = int(resume_metadata.get("epoch") or 0)
        resume_extra = resume_metadata.get("extra", {})
        start_epoch = completed_epoch + 1
        resume_history = resume_extra.get("history")
        best_val_loss = float(resume_extra.get("best_val_loss", float("inf")))
        epochs_without_improvement = int(
            resume_extra.get("epochs_without_improvement", 0)
        )
        resumed_from = str(resume_path)
        print(f"Resume     : {resume_path} (mulai epoch {start_epoch})")

    started_at = datetime.now().astimezone()
    session_started = time.perf_counter()
    history = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        patience=int(training_config.get("patience", 5)),
        checkpoint_path=checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        history_path=history_path,
        scheduler=scheduler,
        scaler=scaler,
        amp_enabled=amp_enabled,
        task_names=task_names,
        grad_clip_norm=float(training_config.get("grad_clip_norm", 1.0)),
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        start_epoch=start_epoch,
        history=resume_history,
        best_val_loss=best_val_loss,
        epochs_without_improvement=epochs_without_improvement,
    )
    finished_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - session_started
    if history["val_metrics"]:
        print_task_metrics(history["val_metrics"][-1])

    if mode in {"pilot", "full"}:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": experiment_config["name"],
            "mode": mode,
            "device": str(device),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "torch_version": torch.__version__,
            "mixed_precision": amp_enabled,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "resumed_from": resumed_from,
            "config": config,
            "history": history,
        }
        history_path.write_text(
            json.dumps(to_json_safe(payload), indent=2),
            encoding="utf-8",
        )
        print(f"Riwayat eksperimen tersimpan: {history_path}")

    print("\nTraining selesai dengan aman.")


if __name__ == "__main__":
    main()
