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
from discrete_diffusion.evaluation.stats import eval_graph_list
from discrete_diffusion.io_utils import load_TU_dataset

pylogger = logging.getLogger(__name__)


def test(cfg: DictConfig):
    gt_dataset, _ = load_TU_dataset([Path(cfg.gt_dataset)], [cfg.gt_dataset_name], output_type="nx")
    for dataset_folder, dataset_name in zip(cfg.generated_dataset_folder_list, cfg.generated_dataset_names):
        dataset, _ = load_TU_dataset([Path(dataset_folder)], [dataset_name], output_type="nx")
        results = eval_graph_list(dataset, gt_dataset)
        wandb.log(
            {"evaluation": results}
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="test_generation")
def main(cfg: omegaconf.DictConfig):
    wandb.init(project="discrete_diffusion", entity="graph_generation")
    test(cfg)


if __name__ == "__main__":
    main()
