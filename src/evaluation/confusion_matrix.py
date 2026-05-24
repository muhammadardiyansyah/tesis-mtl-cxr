import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_name="Class"
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.title(
        f"Confusion Matrix - {class_name}"
    )

    plt.show()