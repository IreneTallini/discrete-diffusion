import logging
import os.path
from typing import Any, Mapping, Optional

import torch
from omegaconf import DictConfig

from discrete_diffusion.data.datamodule import MetaData
from discrete_diffusion.pl_modules.pl_module import TemplatePLModule

pylogger = logging.getLogger(__name__)


class ClassifierPLModule(TemplatePLModule):
    def __init__(self, model: DictConfig, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(model, metadata=metadata, *args, **kwargs)

        self.exists_standard = False
        if os.path.exists(self.hparams.standard_model_ckpt_path):
            standard_model = self.instantiate_model(self.hparams.standard_model, metadata)
            self.standard_model = standard_model.load_from_checkpoint(
                checkpoint_path=self.hparams.standard_model_ckpt_path / "checkpoints/file.ckpt")
            self.exists_standard = True
        else:
            # If checkpoints don't exist assume we are training with standard dataset
            print("Standard dataset checkpoints not present: training model with standard dataset")

    def step(self, batch) -> Mapping[str, Any]:
        logits = self(batch)
        loss_fn = torch.nn.CrossEntropyLoss()
        y = torch.tensor(batch.y)
        loss = loss_fn(input=logits, target=y)
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch)
        logits = step_out["logits"]
        acc = self.compute_accuracy(logits, torch.tensor(batch.y))
        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach(),
             "acc/val": acc.cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch)
        logits = step_out["logits"]
        acc = self.compute_accuracy(logits, torch.tensor(batch.y))
        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach(),
             "acc/test": acc.cpu().detach(),
             }
        )

        if self.exists_standard:
            standard_step_out = self.standard_model.step(batch)
            standard_logits = standard_step_out["logits"]
            standard_acc = self.compute_accuracy(standard_logits, torch.tensor(batch.y))
            self.log_dict(
                {"standard_loss/test": standard_step_out["loss"].cpu().detach(),
                 "standard_acc/test": standard_acc.cpu().detach()},
            )

        return step_out

    @staticmethod
    def compute_accuracy(logits, gt_labels):
        labels_prob = torch.sigmoid(logits)
        _, labels = torch.max(labels_prob, dim=1)
        return torch.abs(labels - gt_labels).float().mean()
