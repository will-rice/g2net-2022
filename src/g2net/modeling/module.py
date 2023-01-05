"""Model"""
from typing import Any, Dict, Optional, Union

import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score
from timm.scheduler import CosineLRScheduler
from torch import Tensor, nn
from torchmetrics import MetricCollection, classification

from src.g2net.data import Batch
from src.g2net.modeling.config import ModelConfig


class BaseLightningModule(LightningModule):
    """Simple Model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters(config._asdict())

        self.config = config

        metrics = MetricCollection(
            classification.BinaryAccuracy(),
            classification.BinaryPrecision(),
            classification.BinaryRecall(),
            classification.BinaryF1Score(),
            classification.BinaryAUROC(average="macro"),
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_auroc = classification.BinaryAUROC(average="macro")

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.pos_weight))
        self._fold = 0

    def forward(self, batch: Batch) -> Tensor:
        """Forward Pass."""
        raise NotImplementedError

    def training_step(
        self, batch: Batch, batch_idx: Any
    ) -> Union[Tensor, Dict[str, Any]]:
        """Train step."""
        logits = self(batch)
        loss = self.loss_fn(logits, batch.targets)

        metrics = self.train_metrics(logits, batch.targets)
        metrics["train_loss"] = loss

        self.log_dict({f"fold_{self._fold}": metrics})

        return loss

    def on_train_epoch_end(self) -> None:
        """Train epoch end."""
        self.train_metrics.reset()

    def validation_step(self, batch: Batch, batch_idx: Any) -> None:
        """Val step."""
        logits = self(batch).detach()
        loss = self.loss_fn(logits, batch.targets)

        self.val_metrics.update(logits, batch.targets.detach())
        self.log_dict({"val_loss": loss, f"fold_{self._fold}": {"fold_val_loss": loss}})

    def on_validation_epoch_end(self) -> None:
        """Val epoch end."""
        output = self.val_metrics.compute()
        self.log_dict({f"fold_{self._fold}": output})

        # reset val metrics to prevent memory leak
        self.val_metrics.reset()

    def test_step(self, batch: Batch, batch_idx: Any) -> None:
        """Test step."""
        logits = self(batch).detach()
        loss = self.loss_fn(logits, batch.targets)
        auroc = self.test_auroc(logits, batch.targets)
        sklearn_auroc = roc_auc_score(batch.targets.cpu(), logits.sigmoid().cpu())

        self.log_dict(
            {
                f"fold_{self._fold}": {
                    "test_loss": loss,
                    "test_auroc": auroc,
                    "sklearn_auroc": sklearn_auroc,
                }
            }
        )

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        """Predict step"""

        probs = 0.0
        for i in range(5):
            logits = self(batch).detach()
            probs += logits.sigmoid() / 5.0

        return {"id": batch.file_id[0], "target": probs.item()}

    def configure_optimizers(self) -> Any:
        """Setup optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        lr_scheduler = {
            "scheduler": CosineLRScheduler(
                optimizer,
                t_initial=self.config.max_epochs - 1,
                warmup_t=self.config.warmup_t,
                lr_min=self.config.lr_min,
                warmup_lr_init=0.0,
                warmup_prefix=True,
            )
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)
