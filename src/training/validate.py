import torch


def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, labels, mask in dataloader:

            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
                mask
            )

            running_loss += loss.item()

    epoch_loss = (
        running_loss / len(dataloader)
    )

    return epoch_loss