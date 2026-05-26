import torch


class EarlyStopping:

    def __init__(
        self,
        patience=3,
        min_delta=0.0,
        save_path=None
    ):

        self.patience = patience

        self.min_delta = min_delta

        self.save_path = save_path

        self.best_loss = None

        self.counter = 0

        self.early_stop = False

    def __call__(
        self,
        val_loss,
        model=None
    ):

        if self.best_loss is None:

            self.best_loss = val_loss

            self.save_best_model(model)

        elif val_loss < (
            self.best_loss - self.min_delta
        ):

            self.best_loss = val_loss

            self.counter = 0

            self.save_best_model(model)

        else:

            self.counter += 1

            print(
                f"EarlyStopping Counter: "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.early_stop = True

    def save_best_model(
        self,
        model
    ):

        if (
            self.save_path is not None
            and model is not None
        ):

            torch.save(
                model.state_dict(),
                self.save_path
            )

            print(
                f"Best model saved to: "
                f"{self.save_path}"
            )