"""Dataset."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import h5py
import pandas as pd
import torch
import torchvision.transforms as VT
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

import src.g2net.transforms as GT

FAST_TRAINING = True


class Batch(NamedTuple):
    """Single sample."""

    file_id: str
    h1_amplitude: Tensor
    l1_amplitude: Tensor
    targets: Tensor


class BaseKFoldDataModule(LightningDataModule, ABC):
    """Base class for KFold DataModules."""

    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


class KFoldDataModule(BaseKFoldDataModule):
    """Datamodule that holds K folds"""

    def __init__(self, root: Path, config: Any) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._root = root
        self._config = config

        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._predict_dataset: Optional[Dataset] = None
        self._train_fold: Optional[Dataset] = None
        self._val_fold: Optional[Dataset] = None
        self.transforms = VT.RandomApply(
            [
                GT.VerticalShift(p=1.0),
                VT.RandomErasing(p=0.5),
                VT.RandomHorizontalFlip(p=0.5),
                VT.RandomVerticalFlip(p=0.5),
                GT.TimeMasking(10, num_masks=1),
                GT.FreqMasking(10, num_masks=2),
            ],
            p=config.augment_p,
        )

    def setup(self, stage: Optional[str] = "fit") -> None:
        """Setup datasets."""

        if stage == "fit":

            self._train_dataset = G2NetDataset(self._root)
            self._test_dataset = G2NetDataset(self._root, split="val")
            self.setup_folds(self._config.num_folds)
            self.setup_fold_index(0)

        if stage == "predict":
            self._predict_dataset = G2NetDataset(
                self._root.parent / "g2net-detecting-continuous-gravitational-waves",
                "test",
            )

    def setup_folds(self, num_folds: int) -> None:
        """Setup folds."""
        self.num_folds = num_folds
        self.splits = [
            split
            for split in KFold(num_folds, random_state=42, shuffle=True).split(
                range(len(self._train_dataset))
            )
        ]

    def setup_fold_index(self, fold_index: int) -> None:
        """Setup fold index."""
        train_indices, val_indices = self.splits[fold_index]
        self._train_fold = Subset(self._train_dataset, train_indices)
        self._val_fold = Subset(self._train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self._train_fold,
            batch_size=self._config.batch_size,
            num_workers=self._config.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Validate dataloader."""
        return DataLoader(
            self._val_fold,
            batch_size=self._config.batch_size,
            num_workers=self._config.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self._test_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Predict dataloader."""
        return DataLoader(
            self._predict_dataset,
            batch_size=1,
            num_workers=self._config.num_workers,
            shuffle=False,
            pin_memory=False,
        )


class G2NetDataset(Dataset):
    """G2net dataset."""

    def __init__(self, root: Path, split: str = "train", transform=None):
        super().__init__()
        self._root = root
        self._split = split
        self._transform = transform

        if split == "test":
            self.samples = pd.read_csv(self._root / "sample_submission.csv")
            self.targets = self.samples["target"].values
        else:
            samples = pd.read_csv(self._root / "train_labels.csv")
            samples = samples.loc[samples["target"] != -1]
            targets = samples["target"].values

            train, test = train_test_split(
                samples, test_size=0.2, random_state=42, stratify=targets
            )
            if split == "train":
                self.samples = train
                self.targets = self.samples["target"].values

                if FAST_TRAINING:
                    # load all samples into memory for faster training
                    print("Loading train samples...")
                    self.samples = [
                        self.load(i) for i in tqdm(range(len(self.samples)))
                    ]
            else:
                self.samples = test
                self.targets = self.samples["target"].values

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        if FAST_TRAINING:
            if self._split == "train":
                return self.samples[idx]

        if isinstance(idx, Sequence):
            return [self.load(i) for i in idx]
        return self.load(idx)

    def load(self, idx):
        """Load sample."""

        sample = self.samples.iloc[idx]
        file_id, target = sample["id"], sample["target"]

        stage = "test" if self._split == "test" else "train"
        with h5py.File(self._root / stage / f"{file_id}.hdf5") as file:

            l1_amplitude = file[file_id]["L1"]["SFTs"][:]
            l1_amplitude = self.preprocess(torch.from_numpy(l1_amplitude))

            h1_amplitude = file[file_id]["H1"]["SFTs"][:]
            h1_amplitude = self.preprocess(torch.from_numpy(h1_amplitude))

            if self._transform:
                l1_amplitude = self._transform(l1_amplitude)
                h1_amplitude = self._transform(h1_amplitude)

            target = torch.tensor([target]).float()

        return Batch(
            file_id=file_id,
            h1_amplitude=h1_amplitude,
            l1_amplitude=l1_amplitude,
            targets=target,
        )

    @staticmethod
    def preprocess(amplitude, mean_pool=True):
        """Preprocess amplitude."""
        amplitude = amplitude[:, :4096] * 1e22
        amplitude = amplitude.real**2 + amplitude.imag**2
        amplitude /= amplitude.mean()

        if mean_pool:
            amplitude = torch.mean(amplitude.reshape(360, 128, 32), dim=2)

        amplitude = amplitude[None]
        return amplitude
