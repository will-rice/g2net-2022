"""Ensemble of models."""
from typing import Any, List, Type

import torch
import torch.nn.functional as F
from pytorch_lightning.core.module import LightningModule
from torchmetrics.classification import BinaryAUROC


class EnsembleVotingModel(LightningModule):
    def __init__(
        self, model_cls: Type[LightningModule], config, checkpoint_paths: List[str]
    ) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(p, config=config) for p in checkpoint_paths]
        )
        self.test_roc = BinaryAUROC(average="macro")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # Compute the averaged predictions over the `num_folds` models.
        logits = torch.stack([m(batch).detach() for m in self.models]).mean(0)
        loss = F.binary_cross_entropy_with_logits(logits, batch.targets)
        self.test_roc(logits, batch.targets)
        self.log("ensemble_roc", self.test_roc)
        self.log("ensemble_loss", loss)

        del logits

        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Predict step"""
        logits = torch.stack([m(batch).detach() for m in self.models]).mean(0)
        probs = torch.sigmoid(logits)

        return {"id": batch.file_id[0], "target": probs.item()}
