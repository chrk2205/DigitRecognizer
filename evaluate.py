import torch
from models.cnn.model import Model
from data.dataloader import TestDataLoader
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm


def generate_submission():
    model = Model()
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()
    x_test = DataLoader(TestDataLoader())
    df = pd.DataFrame(columns=["ImageId", "Label"])
    for idx, x in tqdm(enumerate(x_test)):
        y_test = model(x)
        df = df._append({"ImageId": idx + 1, "Label": int(torch.argmax(y_test))}, ignore_index=True)
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    generate_submission()
