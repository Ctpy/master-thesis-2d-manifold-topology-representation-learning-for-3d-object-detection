from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from typing import Optional
from pathlib import Path
from dataset.kitti_cropped_dataset import KITTICroppedDataset
import numpy as np

class KITTICroppedDataloader(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32) -> None:
        super().__init__()
        self.data_path: Path = Path(data_path)
        self.dataset: KITTICroppedDataset = KITTICroppedDataset(data_path=data_path, min_points=100)
        self.batch_size: int = batch_size

    def prepare_data(self) -> None:
        # Downlaod data if needed
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_size = int(np.ceil(len(self.dataset) * 0.8))
            val_size = int(np.ceil(len(self.dataset) * 0.1))
            test_size = len(self.dataset) - train_size - val_size
            print(f"train_size: {train_size}")
            print(f"val_size: {val_size}")
            print(f"test_size: {test_size}")
            print(f"total: {len(self.dataset)} = {train_size + val_size + test_size}")
            self.train, self.validate, self.test = random_split(
                self.dataset, [train_size, val_size, test_size]
            )
        elif stage == "test" or stage is None:
            test_size = int(len(self.dataset) * 0.1)
            self.test, _ = random_split(self.dataset, [test_size])
        elif stage == "overfit" or stage is None:
            self.overfit, _ = random_split(self.dataset, [1])
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validate, self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, self.batch_size, shuffle=False, num_workers=2)

    def overfit_dataloader(self) -> DataLoader:
        return DataLoader(self.overfit, self.batch_size, shuffle=False, num_workers=2)
