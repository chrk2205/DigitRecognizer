
from torch.utils.data import Dataset
import pandas as pd
import torch

class MNISTDataLoader(Dataset):
    def __init__(self, batch_size=10, file_path="data/train.csv"):
        self.batch_size = batch_size
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def _one_hot(self,digit: int):
        return [1 if i == digit else 0 for i in range(10)]

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float)
        y = torch.tensor(self._one_hot(self.data.iloc[idx, 0]), dtype=torch.float)
        return x, y
