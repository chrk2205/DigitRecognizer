from typing import Type
import torch
import pandas as pd


def convert_digit_to_one_hot(digits: list[int]):
    return [[1 if i == digit else 0 for i in range(10)] for digit in digits]


def load_data(x: Type = torch.float, y: Type = torch.long):
    train = pd.read_csv("data/train.csv")

    # split train and val
    # 80% for train, 20% for val
    train, val = train.iloc[: int(len(train) * 0.8)], train.iloc[int(len(train) * 0.8) :]

    x_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values

    x_val = val.iloc[:, 1:].values
    y_val = val.iloc[:, 0].values

    x_train = torch.tensor(x_train, dtype=x)
    y_train = torch.tensor(convert_digit_to_one_hot(y_train), dtype=y)

    x_val = torch.tensor(x_val, dtype=x)
    y_val = torch.tensor(convert_digit_to_one_hot(y_val), dtype=y)

    return (x_train, y_train), (x_val, y_val)


def load_test_data(x: Type = torch.float):
    test = pd.read_csv("data/test.csv")
    x_test = test.iloc[:, :].values
    x_test = torch.tensor(x_test, dtype=x)
    return x_test


# if __name__ == "__main__":
#     print(convert_digit_to_one_hot([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
