import torch
from model import NeuralNetwork
from data.load_data import load_test_data

import pandas as pd
import matplotlib.pyplot as plt


def evaluate():
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    x_test = load_test_data()
    y_test = model(x_test[0])

    df = pd.read_csv("data/test.csv")

    img = df.iloc[0, :].values.reshape(28, 28)

    plt.imshow(img)
    plt.axis("off")
    plt.savefig("test.png")

    print(y_test)


if __name__ == "__main__":
    evaluate()
