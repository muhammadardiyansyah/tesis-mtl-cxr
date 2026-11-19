"""STL frontal-only (salinan run_single_task_training.py, 17 Sep 2026).

Perbedaan dari skrip asal hanya: split patient_level_frontal_v1, folder
checkpoint/output terpisah, pos_weight kardiomegali 10.1854, dan argumen
--seed. Arsitektur, augmentasi, sampler, optimizer, dan early stopping sama
persis dengan pembanding MTL agar perbandingan MTL vs STL adil.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset_loader import ChestXrayDataset
from src.datasets.source_balanced_sampler import SourceBalancedBatchSampler
from src.datasets.transforms import get_eval_transforms, get_train_transforms
from src.models.multitask_model import MultiTaskModel


RAW_DATA = Path("/kaggle/working/mtl_cxr_data/raw")
SPLIT_DIR = PROJECT_ROOT / "data/splits/patient_level_frontal_v1"
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints_single_task_frontal")
OUTPUT_DIR = Path("/kaggle/working/outputs_single_task_frontal")

TASK_PROTOCOLS = {
    "cardiomegaly": {
        "column": "cardiomegaly",
        "index": 0,
        "pos_weight": 10.1854,  # 187.595 / 18.418 pada train frontal-only
        "samples_per_source": {
            "chexpert": 8,
            "nih": 8,
        },
    },
    "tuberculosis": {
        "column": "tb",
        "index": 1,
        "pos_weight": 9.5000,
        "samples_per_source": {
            "tbx11k": 16,
        },
    },
}

SEED = 42
IMAGE_SIZE = 224
EVAL_BATCH_SIZE = 128
STEPS_PER_EPOCH = 368
MAX_EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ActiveTaskDataset(Dataset):
    def __init__(
        self,
        csv_file: Path,
        task_name: str,
        transform,
    ) -> None:
        protocol = TASK_PROTOCOLS[task_name]

        self.task_name = task_name
        self.label_column = protocol["column"]
        self.task_index = protocol["index"]

        self.base_dataset = ChestXrayDataset(
            csv_file=csv_file,
            raw_data_dir=RAW_DATA,
            transform=transform,
        )

        active = (
            self.base_dataset.dataframe[self.label_column]
            .isin([0, 1])
            .to_numpy()
        )

        self.base_indices = np.flatnonzero(active)
        self.dataframe = (
            self.base_dataset.dataframe
            .iloc[self.base_indices]
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.base_indices)

    def __getitem__(self, index: int):
        base_index = int(self.base_indices[index])
        image, labels, masks = self.base_dataset[base_index]

        target = labels[self.task_index].reshape(1)
        return image, target


def build_loaders(
    task_name: str,
    mode: str,
):
    protocol = TASK_PROTOCOLS[task_name]

    train_dataset = ActiveTaskDataset(
        csv_file=SPLIT_DIR / "train.csv",
        task_name=task_name,
        transform=get_train_transforms(
            image_size=IMAGE_SIZE
        ),
    )

    val_dataset = ActiveTaskDataset(
        csv_file=SPLIT_DIR / "val.csv",
        task_name=task_name,
        transform=get_eval_transforms(
            image_size=IMAGE_SIZE
        ),
    )

    steps = 2 if mode == "smoke" else STEPS_PER_EPOCH

    batch_sampler = SourceBalancedBatchSampler(
        sources=train_dataset.dataframe["source"].tolist(),
        samples_per_source=protocol["samples_per_source"],
        seed=SEED,
        steps_per_epoch=steps,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    common = {
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        **common,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        **common,
    )

    return train_loader, val_loader


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "n": int(len(labels)),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "f1": float(
            f1_score(labels, predictions)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
):
    model.train()

    batch_sampler = getattr(loader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch - 1)

    running_loss = 0.0
    amp_enabled = device.type == "cuda"

    progress = tqdm(
        loader,
        desc="Training",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Training loss menjadi NaN atau inf."
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


@torch.inference_mode()
def validate_epoch(
    model,
    loader,
    criterion,
    device,
    max_batches=None,
):
    model.eval()

    running_loss = 0.0
    processed = 0
    all_labels = []
    all_probabilities = []

    progress = tqdm(
        loader,
        desc="Validation",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for batch_index, (images, targets) in enumerate(progress):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        probabilities = torch.sigmoid(logits)

        running_loss += float(loss.item())
        processed += 1

        all_labels.append(
            targets.cpu().numpy().reshape(-1)
        )
        all_probabilities.append(
            probabilities.float().cpu().numpy().reshape(-1)
        )

        progress.set_postfix(loss=f"{loss.item():.4f}")

    if processed == 0:
        raise RuntimeError(
            "Validation tidak menghasilkan batch."
        )

    labels = np.concatenate(all_labels).astype(int)
    probabilities = np.concatenate(all_probabilities)

    return {
        "loss": running_loss / processed,
        "metrics": binary_metrics(
            labels,
            probabilities,
        ),
    }


def save_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=tuple(TASK_PROTOCOLS),
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Lanjutkan dari checkpoint latest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed eksperimen (42, 43, 44 untuk analisis multi-seed).",
    )
    args = parser.parse_args()

    # Seed dipakai oleh sampler, generator DataLoader, dan nama eksperimen.
    global SEED
    SEED = int(args.seed)
    set_seed(SEED)

    task_name = args.task
    mode = args.mode
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    epochs = 1 if mode == "smoke" else MAX_EPOCHS
    max_val_batches = 2 if mode == "smoke" else None
    experiment_name = f"resnet18_stl_{task_name}_frontal_v1_s{SEED}"

    best_path = (
        CHECKPOINT_DIR /
        f"{experiment_name}_best.pt"
    )
    latest_path = (
        CHECKPOINT_DIR /
        f"{experiment_name}_latest.pt"
    )
    history_path = (
        OUTPUT_DIR /
        f"{experiment_name}_history.json"
    )

    print("=" * 72)
    print("SINGLE-TASK RESNET18 TRAINING")
    print("=" * 72)
    print("Task              :", task_name)
    print("Mode              :", mode)
    print("Device            :", device)
    print("GPU aktif         :", (
        torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else "CPU"
    ))
    print("Pretrained        :", mode == "full" and not args.resume)
    print("Epoch maksimum    :", epochs)
    print("Langkah per epoch :", (
        2 if mode == "smoke"
        else STEPS_PER_EPOCH
    ))
    print("Batch aktif       : 16")
    print("AMP               :", device.type == "cuda")

    train_loader, val_loader = build_loaders(
        task_name,
        mode,
    )

    # Saat resume, bobot pretrained tidak perlu diunduh lagi.
    model = MultiTaskModel(
        backbone_name="resnet18",
        pretrained=(mode == "full" and not args.resume),
        dropout=0.20,
        task_names=(task_name,),
        freeze_backbone=False,
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [TASK_PROTOCOLS[task_name]["pos_weight"]],
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda"),
        )
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(
            enabled=(device.type == "cuda"),
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": [],
        "learning_rate": [],
    }

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume:
        if mode != "full":
            raise ValueError(
                "--resume hanya digunakan bersama --mode full."
            )
        if not latest_path.exists():
            raise FileNotFoundError(
                f"Checkpoint latest tidak ditemukan: {latest_path}"
            )

        checkpoint = torch.load(
            latest_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(
            checkpoint["best_val_loss"]
        )
        epochs_without_improvement = int(
            checkpoint["epochs_without_improvement"]
        )
        history = checkpoint["history"]

        print("Resume dari epoch :", start_epoch)

        if epochs_without_improvement >= PATIENCE:
            print(
                "Training sebelumnya sudah berhenti "
                "melalui early stopping."
            )
            return

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        started = perf_counter()

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
        )

        validation = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            max_batches=max_val_batches,
        )

        val_loss = float(validation["loss"])
        metrics = validation["metrics"]
        elapsed_minutes = (
            perf_counter() - started
        ) / 60

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(metrics)
        history["learning_rate"].append(
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Waktu: {elapsed_minutes:.1f} menit"
        )
        print(
            f"AUC: {metrics['roc_auc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Sens: {metrics['sensitivity']:.4f} | "
            f"Spes: {metrics['specificity']:.4f}"
        )

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(
                "Early stopping: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        scheduler.step(val_loss)

        checkpoint_payload = {
            "experiment": experiment_name,
            "task": task_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement":
                epochs_without_improvement,
            "history": history,
            "protocol": {
                "seed": SEED,
                "image_size": IMAGE_SIZE,
                "steps_per_epoch": STEPS_PER_EPOCH,
                "active_batch_size": 16,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "pos_weight":
                    TASK_PROTOCOLS[task_name]["pos_weight"],
                "samples_per_source":
                    TASK_PROTOCOLS[task_name][
                        "samples_per_source"
                    ],
            },
        }

        if mode == "full":
            if improved:
                save_atomic(
                    checkpoint_payload,
                    best_path,
                )
                print(
                    "Checkpoint best tersimpan:",
                    best_path,
                )

            save_atomic(
                checkpoint_payload,
                latest_path,
            )

            save_json_atomic(
                {
                    "experiment": experiment_name,
                    "task": task_name,
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "protocol":
                        checkpoint_payload["protocol"],
                },
                history_path,
            )

            print(
                "Checkpoint latest tersimpan:",
                latest_path,
            )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping dipicu.")
            break

    print("\nTraining selesai dengan aman.")

    if mode == "full":
        print("Best checkpoint :", best_path)
        print("Latest checkpoint:", latest_path)
        print("History          :", history_path)


if __name__ == "__main__":
    main()
