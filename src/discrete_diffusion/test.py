import logging

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import discrete_diffusion  # noqa

pylogger = logging.getLogger(__name__)


def test(cfg: DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)

    metadata = getattr(datamodule, "metadata", None)

    standard_model_ckpt = NNCheckpointIO().load_checkpoint(path=cfg.standard_model_ckpt)
    augmented_model_ckpt = NNCheckpointIO().load_checkpoint(path=cfg.augmented_model_ckpt)

    standard_model = hydra.utils.instantiate(cfg.module, _recursive_=False, metadata=metadata)
    augmented_model = hydra.utils.instantiate(cfg.module, _recursive_=False, metadata=metadata)

    standard_model.load_state_dict(standard_model_ckpt['state_dict'])
    augmented_model.load_state_dict(augmented_model_ckpt['state_dict'])

    dataloader = datamodule.test_dataloader()[0]

    for batch in dataloader:
        standard_logits = standard_model(batch)
        augmented_logits = augmented_model(batch)

        gt_labels = torch.tensor(batch.y)

        standard_accuracy = standard_model.compute_accuracy(standard_logits, gt_labels)
        augmented_accuracy = augmented_model.compute_accuracy(augmented_logits, gt_labels)

        wandb.log(
            {"standard_accuracy": standard_accuracy,
             "augmented_accuracy": augmented_accuracy}
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="test")
def main(cfg: omegaconf.DictConfig):
    test(cfg)


if __name__ == "__main__":
    main()
