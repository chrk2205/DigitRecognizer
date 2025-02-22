import torch
from model import NeuralNetwork
from train import get_model
from data.dataloader import MNISTTestDataLoader
from torch.utils.data import DataLoader
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

# from PIL import Image
from tqdm import tqdm

def generate_submission():
    model = get_model()
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()

    x_test = DataLoader(MNISTTestDataLoader())

    df = pd.DataFrame(columns=["ImageId", "Label"])


    for idx, x in tqdm(enumerate(x_test)):
        y_test = model(x)
        df = df._append({"ImageId": idx+1, "Label": int(torch.argmax(y_test))}, ignore_index=True)
        # break
        # print(torch.argmax(y_test))
        # # print(x.shape)
        # x = x.squeeze()
        # image = Image.fromarray((x.numpy() * 255).astype('uint8'))
        # image.save(f"{idx}.png")

        # plt.imshow(img)
        # plt.axis("off")
        # plt.savefig(f"{df.iloc[0,0]}.png")

        # break



    df.to_csv("submission.csv", index=False)

    # print(y_test)


if __name__ == "__main__":
    generate_submission()
