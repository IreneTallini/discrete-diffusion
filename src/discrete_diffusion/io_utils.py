import logging
import math
import random
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas
import torch
import torch_geometric.data
from torch import Tensor
from torch_geometric.data import Batch, Data

pylogger = logging.getLogger(__name__)


def load_TU_dataset(paths: List[Path], dataset_names: List[str], output_type="pyg",
                    max_graphs_per_dataset=None, max_num_nodes=-1, iso_test=False):
    if max_graphs_per_dataset is None:
        max_graphs_per_dataset = [-1] * len(paths)
    big_graphs_list = []
    features_list = []
    G = nx.Graph()
    # load d
    # min_labels = 1000
    # for path, dataset_name in zip(paths, dataset_names):
    #     data_graph_labels = np.loadtxt(path / (dataset_name + "_graph_labels.txt"), delimiter=",").astype(int)
    #     min_labels_tmp = min(data_graph_labels)
    #     if min_labels_tmp < min_labels:
    #         min_labels = min_labels_tmp

    for path, dataset_name, dataset_id in zip(paths, dataset_names, range(len(paths))):
        pylogger.info(f"reading dataset from {path}")
        data_adj = np.loadtxt(str(path / (dataset_name + "_A.txt")), delimiter=",").astype(int)
        data_node_att = np.loadtxt(str(path / (dataset_name + "_node_attributes.txt")), delimiter=",")
        data_graph_indicator = np.loadtxt(
            str(path / (dataset_name + "_graph_indicator.txt")), delimiter=",").astype(int)
        data_graph_labels = np.loadtxt(
            str(path / (dataset_name + "_graph_labels.txt")), delimiter=",").astype(int)  # - min_labels + 1

        graphs_list = []

        if max_graphs_per_dataset[dataset_id] == -1 or max_graphs_per_dataset[dataset_id] > len(data_graph_labels):
            graph_num = len(data_graph_labels)
        else:
            graph_num = max_graphs_per_dataset[dataset_id]

        data_tuple = list(map(tuple, data_adj))

        # add edges
        G.add_edges_from(data_tuple)
        # add node attributes
        for idx in range(data_node_att.shape[0]):
            G.add_node(idx + 1, x=torch.tensor(data_node_att[idx]))

        # split into graphs
        node_list = np.arange(data_graph_indicator.shape[0]) + 1
        graphs_list = []
        node_num_list = []
        deck = list(range(1, len(data_graph_labels) + 1))
        random.Random(4).shuffle(deck)
        while (len(graphs_list) < graph_num) and (len(deck) > 0):
            rand_id = deck.pop()
            # find the nodes for each graph
            nodes = node_list[data_graph_indicator == rand_id]
            G_sub = nx.Graph(G.subgraph(nodes))
            if G_sub.number_of_nodes() <= max_num_nodes or max_num_nodes == -1:
                relabeling = {node: n for node, n in zip(sorted(G_sub.nodes), range(1, len(G_sub.nodes) + 1))}
                G_sub = nx.relabel_nodes(G_sub, relabeling)
                G_sub.graph["label"] = data_graph_labels[rand_id - 1]
                graphs_list.append(G_sub)
                node_num_list.append(G_sub.number_of_nodes())
                features_list.append(list(nx.get_node_attributes(G_sub, "x").values()))
        pylogger.info(f"number of graphs: {len(graphs_list)}")
        pylogger.info(f"average number of nodes: {sum(node_num_list) / len(node_num_list)}")

        # Check if the function works
        if iso_test:
            rusty_dataset = torch_geometric.datasets.TUDataset("tmp", "PROTEINS_full")
            iso_count = []
            for idxc, g1 in enumerate(graphs_list):
                for g2 in rusty_dataset:
                    g2 = torch_geometric.utils.to_networkx(g2).to_undirected()
                    if nx.is_isomorphic(g1, g2):
                        iso_count.append(idxc)
                        break
            if len(iso_count) == len(graphs_list):
                pylogger.info("everything isomorphic to loadTUDataset!")
            else:
                pylogger.info("indices of nodes not isomorphic to the ones loaded with loadTUDataset:")
                pylogger.info(iso_count)
        big_graphs_list.extend(graphs_list)

    random.Random(5).shuffle(big_graphs_list)
    if output_type == "pyg":
        big_graphs_list = to_data_list(big_graphs_list)
    return big_graphs_list, features_list


def write_TU_format(graph_list: List[nx.Graph], path: Path, dataset_name):
    pyg_list = to_data_list(graph_list)
    data_batch = Batch.from_data_list(pyg_list)

    path.mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(
        data_batch.edge_index.T.numpy()[:, [1, 0]] + 1, index=list(range(data_batch.num_edges)), columns=["1", "2"]
    ).to_csv(path / (dataset_name + "_A.txt"), header=False, index=False)

    pandas.DataFrame(data_batch.batch.numpy() + 1, index=list(range(data_batch.num_nodes)), columns=[1]).to_csv(
        path / (dataset_name + "_graph_indicator.txt"), header=False, index=False
    )

    pandas.DataFrame(
        data_batch.x.numpy(), index=list(range(data_batch.num_nodes)), columns=list(range(data_batch.x.shape[-1]))
    ).to_csv(path / (dataset_name + "_node_attributes.txt"), header=False, index=False, float_format="%.6f")

    pandas.DataFrame(data_batch.y.numpy() + 1, index=list(range(data_batch.num_graphs)), columns=[1]).to_csv(
        path / (dataset_name + "_graph_labels.txt"), header=False, index=False
    )


def to_data_list(graph_list: List[nx.Graph]) -> List[Data]:
    """
    Converts a list of Networkx graphs to a list of PyG Data objects
    :param graph_list: list of Networkx graphs

    :return: a list of pyg data. Convention: indexing of both nodes, graphs and labels
    in pyg starts from 0, in nx from 1
    """
    data_list_internal = []

    for G in graph_list:
        edge_index = get_edge_index_from_nx(G)

        x = [feat.tolist() for feat in nx.get_node_attributes(G, "x").values()]
        data = Data(
            edge_index=edge_index - 1, num_nodes=G.number_of_nodes(),
            x=torch.tensor(x), y=torch.tensor(G.graph["label"]) - 1
        )

        data_list_internal.append(data)

    return data_list_internal


def get_edge_index_from_nx(g: nx.Graph) -> Tensor:
    """
    Extracts edge index from networkx graph
    :param g: networkx graph
    :return: tensor ~ (2, num_edges) containing all the edges in the graph G
    """
    # shape (num_edges*2, 2)
    edges_tensor = torch.tensor(list([(edge[0], edge[1]) for edge in g.edges]), dtype=torch.long)
    edges_tensor_reverse = torch.tensor(list([(edge[1], edge[0]) for edge in g.edges]), dtype=torch.long)

    edge_index = torch.cat((edges_tensor, edges_tensor_reverse), dim=0)

    return edge_index.t().contiguous()


def random_split_sequence(sequence: List, split_ratio: float) -> Tuple[List, List]:
    f"""
    Splits a sequence randomly into two sequences, the first having {split_ratio}% of the elements
    and the second having {1-split_ratio}%.
    :param sequence: sequence to be split.
    :param split_ratio: percentage of the elements falling in the first sequence.
    :return: subseq_1, subseq_2
    """

    idxs = np.arange(len(sequence))
    np.random.shuffle(idxs)

    support_upperbound = math.ceil(split_ratio * len(sequence))
    split_sequence_1_idxs = idxs[:support_upperbound]
    split_sequence_2_idxs = idxs[support_upperbound:]

    split_seq_1 = [sequence[idx] for idx in split_sequence_1_idxs]
    split_seq_2 = [sequence[idx] for idx in split_sequence_2_idxs]

    return split_seq_1, split_seq_2


def split_sequence(sequence: List, split_ratio: float) -> Tuple[List, List]:
    f"""
    Splits a sequence randomly into two sequences, the first having {split_ratio}% of the elements
    and the second having {1-split_ratio}%.
    :param sequence: sequence to be split.
    :param split_ratio: percentage of the elements falling in the first sequence.
    :return: subseq_1, subseq_2
    """

    support_upperbound = math.ceil(split_ratio * len(sequence))

    split_seq_1 = sequence[:support_upperbound]
    split_seq_2 = sequence[support_upperbound + 1:]

    return split_seq_1, split_seq_2


if __name__ == "__main__":
    data_list, _ = load_TU_dataset(paths=[Path("../../data/standard/PROTEINS_full")], dataset_names=["PROTEINS_full"],
                                   output_type="networkx", max_num_nodes=50)

    random.Random().shuffle(data_list)

    train_end = int(len(data_list) * 0.8)
    val_end = int(len(data_list) * 0.9) - train_end
    data_list_train = data_list[: train_end]
    data_list_val = data_list[train_end + 1: val_end]
    data_list_test = data_list[val_end + 1:]
