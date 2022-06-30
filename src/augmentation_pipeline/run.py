import logging
import math
from typing import List, Optional

import hydra
import networkx as nx
import omegaconf
import pytorch_lightning as pl
import torch_geometric.data
import wandb
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.cm import coolwarm
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import discrete_diffusion  # noqa
from discrete_diffusion.data.datamodule import MetaData
from discrete_diffusion.run import run as run_connectivity_generation
from discrete_diffusion.utils import clear_figures, edge_index_to_adj, generate_sampled_graphs_figures

pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> str:
    # TODO: Load or generate augmented datamodule
    data_root = PROJECT_ROOT / "data" / cfg.connectivity_generation.nn.data.dataset_name
    data_dirs = [data_root / "standard"]
    if cfg.augmentation_pipeline.connectivity.mode in ["train", "restore"]:
        data_dirs.append(data_root / "connectivity")
        if cfg.augmentation_pipeline.connectivity.mode == "train":
            # Override dataset in connectivity_augmented
            run_connectivity_generation(cfg.connectivity_generation)

    if cfg.augmentation_pipeline.node_features.mode in ["train", "restore"]:
        data_dirs.append(data_root / "node_features_augmented")
        if cfg.augmentation_pipeline.node_features.mode == "train":
            # Override dataset in connectivity_augmented
            # train_data(cfg)
            pass

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.connectivity_generation.nn.data,
        data_dirs=data_dirs,
        graph_generator=cfg.connectivity_generation.nn.graph_generator,
        _recursive_=False,
    )

    # TODO: Load classification model
    # TODO: train with standard dataset
    # TODO: train with augmented dataset


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
