"""Deterministic single-image inference for the two MTL outputs."""

from __future__ import annotations

import torch
from PIL import Image

from src.datasets.transforms import get_inference_transforms


TASK_NAMES = ("cardiomegaly", "tuberculosis")


def predict_image(
    model,
    image_path,
    device,
    threshold: float | tuple[float, float] = 0.5,
    image_size: int = 224,
) -> dict:
    """Predict both tasks without random training augmentation."""

    model.eval()

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    transform = get_inference_transforms(image_size=image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0]

    if isinstance(threshold, (int, float)):
        thresholds = torch.tensor(
            [float(threshold), float(threshold)],
            device=probabilities.device,
        )
    else:
        if len(threshold) != 2:
            raise ValueError("threshold must contain two values")
        thresholds = torch.tensor(
            list(threshold),
            dtype=probabilities.dtype,
            device=probabilities.device,
        )

    predictions = probabilities >= thresholds

    return {
        task_name: {
            "probability": float(probabilities[index].cpu()),
            "prediction": int(predictions[index].cpu()),
            "threshold": float(thresholds[index].cpu()),
        }
        for index, task_name in enumerate(TASK_NAMES)
    }
