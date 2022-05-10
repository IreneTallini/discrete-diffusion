from typing import Dict, List

import hydra
import networkx as nx
import numpy as np
import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.data import Data

from nn_core.common import PROJECT_ROOT


class GraphDataset(Dataset):
    def __init__(self, samples: List[Data]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _: Dataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train,
        root=PROJECT_ROOT / "data" / "train",
        cfg=cfg.nn.data.graph_generator,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
