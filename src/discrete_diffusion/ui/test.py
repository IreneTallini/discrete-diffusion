import logging
from pathlib import Path

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from torchmetrics import ConfusionMatrix

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import load_model

# Force the execution of __init__.py if this file is executed directly.
import discrete_diffusion  # noqa

pylogger = logging.getLogger(__name__)


def test(cfg: DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)

    metadata = getattr(datamodule, "metadata", None)

    standard_model = hydra.utils.instantiate(cfg.module, _recursive_=False, metadata=metadata)
    augmented_model = hydra.utils.instantiate(cfg.module, _recursive_=False, metadata=metadata)

    standard_model = load_model(standard_model, checkpoint_path=Path(cfg.standard_model_ckpt))
    augmented_model = load_model(augmented_model, checkpoint_path=Path(cfg.augmented_model_ckpt))

    standard_model.eval()
    augmented_model.eval()

    dataloader = datamodule.test_dataloader()[0]

    standard_mat = torch.zeros((2, 2))
    augmented_mat = torch.zeros((2, 2))

    confmat = ConfusionMatrix(num_classes=2)

    for i, batch in enumerate(dataloader):
        standard_logits = standard_model(batch)
        augmented_logits = augmented_model(batch)

        _, standard_labels = torch.max(torch.sigmoid(standard_logits), dim=1)
        _, augmented_labels = torch.max(torch.sigmoid(augmented_logits), dim=1)

        gt_labels = torch.tensor(batch.y)

        standard_mat += confmat(standard_labels, gt_labels)
        augmented_mat += confmat(augmented_labels, gt_labels)

    wandb.log(
        {"standard confusion matrix": wandb.Table(columns=["class 0", "class 1"], data=standard_mat.tolist()),
         "augmented confusion matrix": wandb.Table(columns=["class 0", "class 1"], data=augmented_mat.tolist())}
    )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="test")
def main(cfg: omegaconf.DictConfig):
    wandb.init(project="discrete_diffusion", entity="graph_generation")
    test(cfg)


if __name__ == "__main__":
    main()
