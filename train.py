from data.dataloader import TrainDataLoader, TestDataLoader

from torch.utils.data import DataLoader, random_split

from trainer import ModelTrainer

from lightning.pytorch import seed_everything
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

seed_everything(42, workers=True)


def train():
    dataloader = TrainDataLoader()
    train_dataloader, val_dataloader = random_split(dataloader, [0.8, 0.2])

    train_dataloader = DataLoader(
        train_dataloader, shuffle=True, num_workers=6, persistent_workers=True
    )
    val_dataloader = DataLoader(val_dataloader, num_workers=6, persistent_workers=True)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    logger = CSVLogger("logs")

    model = ModelTrainer()
    trainer = L.Trainer(
        deterministic=True,
        max_epochs=10,
        callbacks=[early_stopping],
        logger=logger,
    )
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )


if __name__ == "__main__":
    train()
