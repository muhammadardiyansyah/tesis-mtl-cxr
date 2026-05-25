import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torchvision.transforms import Compose

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from src.datasets.transforms import (
    get_train_transforms
)


def generate_gradcam(
    model,
    image_path,
    target_layer,
    device
):

    model.eval()

    image = Image.open(
        image_path
    ).convert("RGB")

    transform = get_train_transforms()

    input_tensor = transform(image)

    image_resized = image.resize(
        (224, 224)
    )

    rgb_image = (
        np.array(image_resized) / 255.0
    )

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    plt.figure(figsize=(8,8))

    plt.imshow(visualization)
    plt.axis("off")

    plt.title("Grad-CAM Visualization")

    plt.show()