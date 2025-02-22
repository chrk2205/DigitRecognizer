import torch
from models.cnn.model import Model
from data.dataloader import TestDataLoader
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from trainer import ModelTrainer


def generate_submission():
    model = Model()
    model.load_state_dict(
        torch.load("./logs/lightning_logs/version_0/checkpoints/model.pth", weights_only=True)
    )
    model.eval()
    x_test = DataLoader(TestDataLoader())
    df = pd.DataFrame(columns=["ImageId", "Label"])
    for idx, x in tqdm(enumerate(x_test)):
        y_test = model(x)
        df = df._append({"ImageId": idx + 1, "Label": int(torch.argmax(y_test))}, ignore_index=True)
    df.to_csv("submission.csv", index=False)


def generate_submission_2():
    model = ModelTrainer.load_from_checkpoint(
        "./logs/lightning_logs/version_0/checkpoints/epoch=21-step=5786.ckpt"
    )
    # model.load_state_dict(torch.load("./logs/lightning_logs/version_0/checkpoints/epoch=21-step=5786.ckpt", weights_only=True))
    model.eval()
    x_test = DataLoader(TestDataLoader())
    df = pd.DataFrame(columns=["ImageId", "Label"])
    for idx, x in tqdm(enumerate(x_test)):
        x = x.to(model.device)
        y_test = model(x)
        preds = torch.argmax(y_test, dim=1)
        df = df._append({"ImageId": idx + 1, "Label": int(preds)}, ignore_index=True)
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    generate_submission_2()
