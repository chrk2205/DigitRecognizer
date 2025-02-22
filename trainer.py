import torch
import torch.nn as nn
from models.cnn.model import Model
import lightning as L


class ModelTrainer(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return self.optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss
