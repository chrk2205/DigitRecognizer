from torch.utils.data import Dataset
import pandas as pd
import torch


class TrainDataLoader(Dataset):
    def __init__(self, batch_size=10, file_path="data/train.csv"):
        self.batch_size = batch_size
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def _one_hot(self, digit: int):
        return [1 if i == digit else 0 for i in range(10)]

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float)
        x = x.view(28, 28).unsqueeze(0)  # 784 -> 1, 28, 28
        y = torch.tensor(self._one_hot(self.data.iloc[idx, 0]), dtype=torch.float)
        return x, y


class TestDataLoader(Dataset):
    def __init__(self, file_path="data/test.csv"):
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx, :].values, dtype=torch.float)
        x = x.view(28, 28).unsqueeze(0)  # 784 -> 1, 28, 28
        return x
