import torch
from PIL import Image

from src.datasets.transforms import (
    get_train_transforms
)


def predict_image(
    model,
    image_path,
    device,
    threshold=0.5
):

    model.eval()

    image = Image.open(
        image_path
    ).convert("RGB")

    transform = get_train_transforms()

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probs = torch.sigmoid(
            outputs
        )

        preds = (
            probs > threshold
        ).float()

    return (
        probs.cpu().numpy(),
        preds.cpu().numpy()
    )