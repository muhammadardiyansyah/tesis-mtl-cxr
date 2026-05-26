import torch
from tqdm import tqdm

from src.training.early_stopping import (
    EarlyStopping
)
from src.training.validate import (
    validate_one_epoch
)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for images, labels, mask in tqdm(
        dataloader,
        desc="Training",
        leave=False
    ):

        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
            mask
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = (
        running_loss / len(dataloader)
    )

    return epoch_loss

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    patience=3,
    checkpoint_path=None
):

    history = {
        "train_loss": [],
        "val_loss": []
    }

    early_stopping = EarlyStopping(
        patience=patience,
        save_path=checkpoint_path
    )

    for epoch in range(epochs):
        print(
            f"\nEpoch "
            f"{epoch+1}/{epochs}"
        )

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        early_stopping(
            val_loss,
            model
        )

        if early_stopping.early_stop:
            print(
                "EARLY STOPPING TRIGGERED"
            )
            break

    return history