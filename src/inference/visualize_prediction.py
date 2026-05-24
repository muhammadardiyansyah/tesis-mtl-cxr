import matplotlib.pyplot as plt
from PIL import Image


def visualize_prediction(
    image_path,
    probs,
    preds,
    class_names
):

    image = Image.open(
        image_path
    ).convert("RGB")

    plt.figure(figsize=(6,6))

    plt.imshow(image)
    plt.axis("off")

    title = ""

    for i in range(len(class_names)):

        pred_label = (
            "Positive"
            if preds[0][i] == 1
            else "Negative"
        )

        prob = probs[0][i]

        title += (
            f"{class_names[i]}: "
            f"{pred_label} "
            f"({prob:.2f})\n"
        )

    plt.title(title)
    plt.tight_layout()

    plt.show()