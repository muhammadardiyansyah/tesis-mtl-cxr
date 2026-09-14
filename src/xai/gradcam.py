"""Task-specific Grad-CAM for the MTL chest X-ray model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.datasets.transforms import PadToSquare, get_inference_transforms


TASK_NAMES = ("cardiomegaly", "tuberculosis")


def generate_gradcam(
    model,
    image_path,
    target_layer,
    device,
    task_index: int,
    image_size: int = 224,
    show: bool = True,
) -> np.ndarray:
    """Generate Grad-CAM for one explicitly selected output task."""

    if task_index not in (0, 1):
        raise ValueError(
            "task_index must be 0 (cardiomegaly) or 1 (tuberculosis)"
        )

    model.eval()

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    transform = get_inference_transforms(image_size=image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    display_image = PadToSquare(fill=0)(image)
    display_image = display_image.resize((image_size, image_size))
    rgb_image = np.asarray(display_image, dtype=np.float32) / 255.0

    targets = [ClassifierOutputTarget(task_index)]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )

    if show:
        plt.figure(figsize=(7, 7))
        plt.imshow(visualization)
        plt.axis("off")
        plt.title(f"Grad-CAM — {TASK_NAMES[task_index]}")
        plt.tight_layout()
        plt.show()

    return visualization
