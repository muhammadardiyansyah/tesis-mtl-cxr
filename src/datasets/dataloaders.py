"""Build reproducible DataLoaders for the MTL chest X-ray project."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.dataset_loader import ChestXrayDataset
from src.datasets.source_balanced_sampler import SourceBalancedBatchSampler
from src.datasets.transforms import get_eval_transforms, get_train_transforms
from src.utils.config import get_config_path, load_config


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy independently inside each DataLoader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(config: dict | None = None) -> dict[str, DataLoader]:
    """Create train, validation, and test DataLoaders from project config."""

    if config is None:
        config = load_config()

    training_config = config.get("training", {})
    sampling_config = config.get("batch_sampling", {})

    image_size = int(training_config.get("image_size", 224))
    random_seed = int(training_config.get("random_seed", 42))
    num_workers = int(training_config.get("num_workers", 0))
    eval_batch_size = int(training_config.get("eval_batch_size", 64))
    pin_memory = bool(training_config.get("pin_memory", torch.cuda.is_available()))

    raw_data_dir = get_config_path(config, "raw_data")

    train_dataset = ChestXrayDataset(
        csv_file=get_config_path(config, "train_csv"),
        transform=get_train_transforms(image_size=image_size),
        raw_data_dir=raw_data_dir,
    )
    val_dataset = ChestXrayDataset(
        csv_file=get_config_path(config, "val_csv"),
        transform=get_eval_transforms(image_size=image_size),
        raw_data_dir=raw_data_dir,
    )
    test_dataset = ChestXrayDataset(
        csv_file=get_config_path(config, "test_csv"),
        transform=get_eval_transforms(image_size=image_size),
        raw_data_dir=raw_data_dir,
    )

    samples_per_source = sampling_config.get(
        "samples_per_source",
        {"nih": 8, "chexpert": 8, "tbx11k": 16},
    )
    configured_steps = sampling_config.get("steps_per_epoch")

    train_batch_sampler = SourceBalancedBatchSampler(
        sources=train_dataset.dataframe["source"].tolist(),
        samples_per_source=samples_per_source,
        seed=random_seed,
        steps_per_epoch=(
            None if configured_steps is None else int(configured_steps)
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    common_loader_options = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        **common_loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **common_loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **common_loader_options,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
