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
            graph: nx.Graph = nx.relabel.convert_node_labels_to_integers(graph)

            for i in range(graph.number_of_nodes()):
                graph.nodes[i]['x'] = 1

            generated_graphs.append(graph)

            if save_path:
                with open(f"{save_path}/{self.graph_type}_{i}.torch", "wb") as f:
                    torch.save(obj=graph, f=f)

        return generated_graphs
