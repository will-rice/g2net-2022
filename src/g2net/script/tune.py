"""Train script."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

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

DEFAULT_MODEL = "transformer"


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
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--log_path", default="logs", type=Path)
    parser.add_argument("--checkpoint_path", default=None, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=str)

    args = parser.parse_args()

    seed_everything(args.seed)

    name = f"{args.model}-{args.name}"
    log_path = args.log_path / name
    log_path.mkdir(exist_ok=True, parents=True)

    config = ModelConfig()

    if args.model == "transformer":
        model = TransformerModel(config)
    elif args.model == "efficientnet":
        model = EfficientNetModel(config)
    elif args.model == "base":
        model = BaseModel(config)
    elif args.model == "unet":
        model = UNetModel(config)
    else:
        raise ValueError(f"Unknown model {args.model}")

    datamodule = KFoldDataModule(args.data_path, batch_size=args.batch_size)
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
    swa_callback = StochasticWeightAveraging(config.swa_learning_rate)

    pretrained = args.checkpoint_path
    last_checkpoint = pretrained if pretrained else log_path / "last.ckpt"

    trainer = Trainer(
        default_root_dir=log_path,
        max_epochs=1,
        accelerator="auto",
        devices=args.num_devices,
        logger=logger,
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, swa_callback],
        gradient_clip_val=1.0,
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

    predict_model = EnsembleVotingModel(model, folds)
    predictions = trainer.predict(predict_model, datamodule=datamodule)

    df = pd.DataFrame.from_records(predictions)
    df.to_csv(log_path / f"{name}-submission.csv", index=False)
    print(df.head())

    df = pd.read_csv(log_path / f"{name}-submission.csv")
    print(df.head())


if __name__ == "__main__":
    main()
