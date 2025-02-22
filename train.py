import torch

from data.dataloader import TrainDataLoader

from models.cnn.model import Model
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

mps_device = torch.device("mps")


def train_model_one_epoch_batch(model, optimizer, criterion, dataloader):
    model.train()
    train_loss = 0
    for idx, (x, y) in enumerate(dataloader):
        x = x.to(mps_device)
        y = y.to(mps_device)
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
        x = x.to(mps_device)
        y = y.to(mps_device)
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
    dataloader = TrainDataLoader()
    train_dataloader, val_dataloader = random_split(dataloader, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataloader)
    val_dataloader = DataLoader(val_dataloader)

    model = Model()
    model.to(mps_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # evaluate the model before training
    val_loss, accuracy = evaluate_model(model, criterion, val_dataloader)
    print(f"Initial val loss: {val_loss}, accuracy: {accuracy}")

    # TODO Early Stopping

    for epoch in range(10):
        train_loss = train_model_one_epoch_batch(model, optimizer, criterion, train_dataloader)
        print(f"Epoch {epoch} train loss: {train_loss}")

        val_loss, accuracy = evaluate_model(model, criterion, val_dataloader)
        print(f"Epoch {epoch} val loss: {val_loss}, accuracy: {accuracy}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved")


if __name__ == "__main__":
    train()
