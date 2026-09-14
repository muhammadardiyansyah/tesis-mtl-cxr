"""Deterministic and training transforms for chest X-ray images."""

from __future__ import annotations

from typing import Union

from PIL import Image, ImageOps
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PadToSquare:
    """Pad an image to a square without distorting its anatomical ratio."""

    def __init__(self, fill: Union[int, tuple[int, int, int]] = 0):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        maximum_side = max(width, height)

        horizontal_padding = maximum_side - width
        vertical_padding = maximum_side - height

        left = horizontal_padding // 2
        right = horizontal_padding - left
        top = vertical_padding // 2
        bottom = vertical_padding - top

        return ImageOps.expand(
            image,
            border=(left, top, right, bottom),
            fill=self.fill,
        )


def get_train_transforms(image_size: int = 224):
    """Return conservative stochastic augmentation for training only.

    Horizontal flipping is intentionally excluded because it reverses
    anatomical laterality and laterality markers. Padding is performed before
    resizing so the cardiothoracic geometry is not stretched into a square.
    """

    return transforms.Compose(
        [
            PadToSquare(fill=0),
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                fill=0,
            ),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_eval_transforms(image_size: int = 224):
    """Return deterministic transforms for validation and testing."""

    return transforms.Compose(
        [
            PadToSquare(fill=0),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_inference_transforms(image_size: int = 224):
    """Inference must use the same deterministic pipeline as evaluation."""

    return get_eval_transforms(image_size=image_size)


# Backward-compatible name for notebooks that used get_val_transforms().
def get_val_transforms(image_size: int = 224):
    return get_eval_transforms(image_size=image_size)
