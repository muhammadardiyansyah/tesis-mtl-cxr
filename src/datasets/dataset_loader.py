import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None
    ):

        self.dataframe = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image_path = row["image_path"]

        image = Image.open(image_path).convert("RGB")

        cardiomegaly = row["cardiomegaly"]
        tb = row["tb"]

        labels = np.array([
            cardiomegaly,
            tb
        ], dtype=float)

        labels[labels < 0] = np.nan

        mask = ~np.isnan(labels)

        labels = np.nan_to_num(
            labels,
            nan=0.0
        )

        labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        if self.transform:

            image = self.transform(image)

        return image, labels, mask