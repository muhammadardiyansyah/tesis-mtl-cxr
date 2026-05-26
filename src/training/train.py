import torch


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for images, labels, mask in dataloader:

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