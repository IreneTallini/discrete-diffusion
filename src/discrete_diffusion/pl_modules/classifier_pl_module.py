import logging
from typing import Any, Mapping

import torch

from discrete_diffusion.pl_modules.pl_module import TemplatePLModule

pylogger = logging.getLogger(__name__)


class ClassifierPLModule(TemplatePLModule):
    def step(self, batch) -> Mapping[str, Any]:
        logits = self(batch)
        loss_fn = torch.nn.CrossEntropyLoss()
        y = torch.tensor(batch.y).type_as(logits).long()
        loss = loss_fn(input=logits, target=y)
        return {"loss": loss, "logits": logits.detach()}

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch)
        logits = step_out["logits"]
        acc = self.compute_accuracy(logits, torch.tensor(batch.y).type_as(logits).long())
        self.log_dict(
            {
                "loss/val": step_out["loss"].cpu().detach(),
                "acc/val": acc.cpu().detach()
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch)
        logits = step_out["logits"]
        acc = self.compute_accuracy(logits, torch.tensor(batch.y).type_as(logits).long())
        self.log_dict(
            {
                "loss/test": step_out["loss"].cpu().detach(),
                "acc/test": acc.cpu().detach(),
            }
        )
        return step_out

    @staticmethod
    def compute_accuracy(logits, gt_labels):
        labels_prob = torch.sigmoid(logits)
        _, labels = torch.max(labels_prob, dim=1)
        return torch.sum(torch.tensor(labels == gt_labels)) / len(labels)
