import numpy as np
import pandas as pd

import torch

from torch.utils.data import Dataset

from PIL import Image


class SingleTaskDataset(Dataset):

    def __init__(
        self,
        csv_file,
        target_label,
        transform=None
    ):

        self.dataframe = pd.read_csv(csv_file)

        self.target_label = target_label

        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image_path = row["image_path"]

        image = Image.open(
            image_path
        ).convert("RGB")

        label = row[self.target_label]

        if label < 0:

            label = np.nan

        mask = ~np.isnan(label)

        if np.isnan(label):

            label = 0.0

        label = torch.tensor(
            [label],
            dtype=torch.float32
        )

        mask = torch.tensor(
            [mask],
            dtype=torch.float32
        )

        if self.transform:

            image = self.transform(image)

        return image, label, mask