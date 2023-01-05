"""Train script."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)

from src.g2net.data import KFoldDataModule
from src.g2net.kfold import KFoldLoop
from src.g2net.modeling import (
    BaseModel,
    EfficientNetModel,
    EnsembleVotingModel,
    ModelConfig,
    TransformerModel,
    UNetModel,
)
from src.g2net.modeling.base2d import BaseModel2d

DEFAULT_MODEL = "efficientnet"


def main() -> None:
    """Main"""

    parser = argparse.ArgumentParser("train parser")
    parser.add_argument("name", type=str)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--project", default="g2net-2022", type=str)
    parser.add_argument(
        "--num_devices", default=1 if torch.cuda.is_available() else None
    )
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--log_path", default="logs", type=Path)
    parser.add_argument("--checkpoint_path", default=None, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=str)

    args = parser.parse_args()

    name = f"{args.model}-{args.name}"
    log_path = args.log_path / name
    log_path.mkdir(exist_ok=True, parents=True)

    config = ModelConfig()

    seed_everything(config.seed)

    if args.model == "transformer":
        model = TransformerModel(config)
    elif args.model == "efficientnet":
        model = EfficientNetModel(config)
    elif args.model == "base":
        model = BaseModel(config)
    elif args.model == "unet":
        model = UNetModel(config)
    elif args.model == "base2d":
        model = BaseModel2d(config)
    else:
        raise ValueError(f"Unknown model {args.model}")

    datamodule = KFoldDataModule(args.data_path, config)
    logger = loggers.WandbLogger(
        project=args.project,
        save_dir=log_path,
        name=name,
        offline=args.debug,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename="{epoch}-{val_loss:.4f}",
        save_last=True,
        monitor="val_loss",
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    pretrained = args.checkpoint_path
    last_checkpoint = pretrained if pretrained else log_path / "last.ckpt"

    trainer = Trainer(
        default_root_dir=log_path,
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=args.num_devices,
        logger=logger,
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, lr_callback],
        gradient_clip_val=config.clip_norm,
        fast_dev_run=args.debug,
    )
    logger.watch(model, log_graph=False)

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(config.num_folds, export_path=log_path)
    trainer.fit_loop.connect(internal_fit_loop)

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=last_checkpoint if last_checkpoint.exists() else None,
    )

    folds = log_path.glob("*.pt")

    predict_model = EnsembleVotingModel(model, config, folds)
    predictions = trainer.predict(predict_model, datamodule=datamodule)

    df = pd.DataFrame.from_records(predictions)

    if df["target"].isna().any():

        for file_id in df["id"].loc[df["target"].isna()].values:
            print(f"NaNs in prediction for {file_id}")

        df = df.fillna(0.0)

    df.to_csv(log_path / f"{name}-submission.csv", index=False)
    print(df.head())

    df = pd.read_csv(log_path / f"{name}-submission.csv")
    print(df.head())


if __name__ == "__main__":
    main()
