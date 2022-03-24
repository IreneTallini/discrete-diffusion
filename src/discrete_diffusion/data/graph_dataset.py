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


class GeneratedGraphDataset(Dataset):
    def __init__(self, samples: List[Data]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GraphGenerator:
    def __init__(self, graph_type: str, nx_generator: DictConfig, num_samples: int = 100, nx_params: Dict = None):
        self.nx_generator = nx_generator
        self.nx_params = nx_params
        self.graph_type = graph_type
        self.num_samples = num_samples

    def generate_data(self, save_path=None):
        """

        :param save_path:
        :return:
        """
        generated_graphs = []

        for i in range(self.num_samples):
            params = {}

            for k, v_list in self.nx_params.items():
                params[k] = np.random.choice(v_list)

            graph = instantiate(self.nx_generator, **params)
            graph = nx.relabel.convert_node_labels_to_integers(graph)

            generated_graphs.append(graph)

            if save_path:
                with open(f"{save_path}/{self.graph_type}_{i}.torch", "wb") as f:
                    torch.save(obj=graph, f=f)

        return generated_graphs


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
