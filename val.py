import torch


def validate(model, criterion, x_val, y_val):
    model.eval()
    with torch.no_grad():
        output = model(x_val)
        loss = criterion(output, y_val)
        return loss.item()
