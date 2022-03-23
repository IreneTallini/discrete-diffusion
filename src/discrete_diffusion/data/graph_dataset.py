import json
import os

import hydra
import networkx as nx
import numpy as np
import omegaconf
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torchvision.datasets import FashionMNIST

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

# Example Dataset from the template


class GeneratedGraphDataset(InMemoryDataset):
    def __init__(self, split, path, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        super().__init__(path, transform, pre_transform, pre_filter)
        data, slices = torch.load(self.processed_paths[0])
        self.l_train = int(len(data) * 0.9)
        if split == "train":
            self.data, self.slices = data[: self.l_train], slices[: self.l_train]
        elif split == "test":
            self.data, self.slices = data[self.l_train :], slices[self.l_train :]

    @property
    def raw_file_names(self):
        return [f"{self.config.data.dataset}.torch"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        gen_graph_list(
            graph_type=self.config.data.dataset,
            possible_params_dict=dict(self.config.data.params),
            corrupt_func=None,
            length=self.config.data.dataset_length,
            save_dir=os.path.join(self.raw_dir),
            file_name=self.config.data.dataset,
        )

    def process(self):
        # Read data into huge `Data` list.
        graph_list = torch.load(os.path.join(self.raw_dir, f"{self.config.data.dataset}.torch"))
        # fig = plot_graphs_list(graph_list, title="dataset_sample", save_dir=self.config.data.data_folder)
        # wandb.log({"dataset sample": wandb.Image(fig)})
        data_list = [
            Data(edge_index=torch.tensor(np.array(g.edges).T, dtype=torch.int64, device=self.config.device))
            for g in graph_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __len__(self):
        return self.l_train


class GraphGenerator:
    def __init__(self, graph_type="grid", possible_params_dict=None, corrupt_func=None):
        if possible_params_dict is None:
            possible_params_dict = {}
        assert isinstance(possible_params_dict, dict)
        self.count = {k: 0 for k in possible_params_dict}
        self.possible_params = possible_params_dict
        self.corrupt_func = corrupt_func
        self.nx_generator = NAME_TO_NX_GENERATOR[graph_type]

    def __call__(self):
        params = {}
        for k, v_list in self.possible_params.items():
            params[k] = np.random.choice(v_list)
        graph = self.nx_generator(**params)
        graph = nx.relabel.convert_node_labels_to_integers(graph)
        if self.corrupt_func is not None:
            graph = self.corrupt_func(self.corrupt_func)
        return graph


NAME_TO_NX_GENERATOR = {
    "grid": nx.generators.grid_2d_graph,  # grid_2d_graph(m, n, periodic=False, create_using=None)
    "gnp": nx.generators.fast_gnp_random_graph,  # fast_gnp_random_graph(n, p, seed=None, directed=False)
    "ba": nx.generators.barabasi_albert_graph,  # barabasi_albert_graph(n, m, seed=None)
    "pow_law": lambda **kwargs: nx.configuration_model(
        nx.generators.random_powerlaw_tree_sequence(**kwargs, gamma=3, tries=2000)
    ),
    "workdir_balanced_tree": nx.generators.balanced_tree,
    "rand_tree": nx.generators.random_tree,
    "except_deg": lambda **kwargs: nx.expected_degree_graph(**kwargs, selfloops=False),
    "cycle": nx.cycle_graph,
    "c_l": nx.circular_ladder_graph,
    "lobster": nx.random_lobster,
    "ego": nx.generators.ego_graph,  # ego_graph(G, n, radius=1, center=True, undirected=False, distance=None)
}


def gen_graph_list(
    graph_type="grid",
    possible_params_dict=None,
    corrupt_func=None,
    length=1024,
    save_dir=None,
    file_name=None,
    max_node=None,
    min_node=None,
):
    params = locals()
    if file_name is None:
        file_name = graph_type + "_" + str(length)
    file_path = os.path.join(save_dir, file_name)
    graph_generator = GraphGenerator(
        graph_type=graph_type, possible_params_dict=possible_params_dict, corrupt_func=corrupt_func
    )
    graph_list = []
    i = 0
    max_N = 0
    while i < length:
        graph = graph_generator()
        if max_node is not None and graph.number_of_nodes() > max_node:
            continue
        if min_node is not None and graph.number_of_nodes() < min_node:
            continue
        # print(i, graph.number_of_nodes(), graph.number_of_edges())
        max_N = max(max_N, graph.number_of_nodes())
        if graph.number_of_nodes() <= 1:
            continue
        graph_list.append(graph)
        i += 1
    if save_dir is not None:
        with open(file_path + ".torch", "wb") as f:
            torch.save(obj=graph_list, f=f)
        with open(file_path + ".txt", "w") as f:
            f.write(json.dumps(params))
            f.write(f"max node number: {max_N}")
    print("max node number: ", max_N)
    return graph_list


class GraphDataset(Dataset):
    def __init__(self, split: Split, **kwargs):
        super().__init__()
        self.split: Split = split

        # example
        self.mnist = FashionMNIST(
            kwargs["path"],
            train=split == "train",
            download=True,
            transform=kwargs["transform"],
        )

    @property
    def class_vocab(self):
        return self.mnist.class_to_idx

    def __len__(self) -> int:
        # example
        return len(self.mnist)

    def __getitem__(self, index: int):
        # example
        return self.mnist[index]

    def __repr__(self) -> str:
        return f"MyDataset({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _: Dataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, split="train", _recursive_=False)


if __name__ == "__main__":
    main()
