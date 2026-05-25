import torch


def save_checkpoint(
    model,
    save_path
):

    torch.save(
        model.state_dict(),
        save_path
    )

    print(
        f"Model saved to: {save_path}"
    )


def load_checkpoint(
    model,
    checkpoint_path,
    device
):

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model.eval()

    print(
        f"Model loaded from: {checkpoint_path}"
    )

    return model