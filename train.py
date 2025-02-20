import torch

from data.dataloader import MNISTDataLoader
from model import NeuralNetwork, MNISTConvClassifier
from torch.optim import SGD
import torch.nn as nn


def train_model_one_epoch_batch(model, optimizer, criterion, dataloader):
    model.train()
    train_loss = 0
    for idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= dataloader.__len__()
    return train_loss


def evaluate_model(model, criterion, dataloader):
    model.eval()
    val_loss = 0
    hits = 0
    misses = 0
    for idx, (x, y) in enumerate(dataloader):
        output = model(x)
        loss = criterion(output, y)
        val_loss += loss.item()
        if torch.argmax(output) == torch.argmax(y):
            hits += 1
        else:
            misses += 1

    val_loss /= dataloader.__len__()
    accuracy = hits / (hits + misses)
    return val_loss, accuracy




def train():

    dataloader = MNISTDataLoader(batch_size=10)

    train_dataloader, val_dataloader = torch.utils.data.random_split(dataloader, [0.8, 0.2])

    # model = NeuralNetwork()

    model = MNISTConvClassifier(
        input_channels=28*28,
        hidden_channels=256,
        num_classes=10,
        hidden_dim=25
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # evaluate the model before training
    val_loss, accuracy = evaluate_model(model, criterion, val_dataloader)
    print(f"Initial val loss: {val_loss}, accuracy: {accuracy}")

    for epoch in range(1):
        train_loss = train_model_one_epoch_batch(model, optimizer, criterion, train_dataloader)
        print(f"Epoch {epoch} train loss: {train_loss}")

        val_loss, accuracy = evaluate_model(model, criterion, val_dataloader)
        print(f"Epoch {epoch} val loss: {val_loss}, accuracy: {accuracy}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved")


if __name__ == "__main__":
    train()
