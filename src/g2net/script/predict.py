"""Train script."""
import argparse
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl

from src.g2net.data import KFoldDataModule
from src.g2net.modeling.efficientnet.model import EfficientNetModel


def main() -> None:
    """Main"""

    parser = argparse.ArgumentParser("train parser")
    parser.add_argument("name", type=str)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("--seed", default=21, type=int)
    parser.add_argument("--log_path", default="logs", type=Path)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    log_path = args.log_path / args.name
    log_path.mkdir(exist_ok=True)

    model = EfficientNetModel()

    datamodule = KFoldDataModule(args.data_path, batch_size=1)

    trainer = pl.Trainer(accelerator="auto")

    model = model.load_from_checkpoint(args.ckpt_path)
    model.eval()

    predictions = trainer.predict(model, datamodule=datamodule)

    df = pd.DataFrame.from_records(predictions)
    df.to_csv(log_path / f"{args.name}-{args.ckpt_path.stem}.csv", index=False)
    print(df.head())

    df = pd.read_csv(log_path / f"{args.name}-{args.ckpt_path.stem}.csv")
    print(df.head())


if __name__ == "__main__":
    main()
