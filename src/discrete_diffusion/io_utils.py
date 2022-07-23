import logging
import math
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx

pylogger = logging.getLogger(__name__)


def load_TU_dataset(paths: List[Path], dataset_names: List[str], output_type="pyg", max_graphs_per_dataset=None):
    graphs_list = []
    features_list = []
    G = nx.Graph()
    # load d
    min_labels = 1000
    for path, dataset_name in zip(paths, dataset_names):
        data_graph_labels = np.loadtxt(path / (dataset_name + "_graph_labels.txt"), delimiter=",").astype(int)
        min_labels_tmp = min(data_graph_labels)
        if min_labels_tmp < min_labels:
            min_labels = min_labels_tmp

    for path, dataset_name, dataset_id in zip(paths, dataset_names, range(len(paths))):
        pylogger.info(f"reading dataset from {path}")
        data_adj = np.loadtxt(path / (dataset_name + "_A.txt"), delimiter=",").astype(int)
        data_node_att = np.loadtxt(path / (dataset_name + "_node_attributes.txt"), delimiter=",")
        data_graph_indicator = np.loadtxt(path / (dataset_name + "_graph_indicator.txt"), delimiter=",").astype(int)
        data_graph_labels = (
            np.loadtxt(path / (dataset_name + "_graph_labels.txt"), delimiter=",").astype(int) - min_labels
        )

        if max_graphs_per_dataset[0] is None:
            graph_num = len(data_graph_labels)
        else:
            assert max_graphs_per_dataset[dataset_id] <= len(data_graph_labels)
            graph_num = max_graphs_per_dataset[dataset_id]

        data_tuple = list(map(tuple, data_adj))

        # add edges
        G.add_edges_from(data_tuple)
        # add node attributes
        for idx in range(data_node_att.shape[0]):
            G.add_node(idx + 1, x=torch.tensor(data_node_att[idx]))

        # split into graphs
        node_list = np.arange(data_graph_indicator.shape[0]) + 1
        graphs = []
        node_num_list = []
        random_ids = torch.randint(0, len(data_graph_labels), size=(graph_num,)).tolist()
        for idx in random_ids:
            # find the nodes for each graph
            nodes = node_list[data_graph_indicator == idx + 1]
            G_sub = nx.Graph(G.subgraph(nodes))
            relabeling = {node: n for node, n in zip(G_sub.nodes, range(1, len(G_sub.nodes) + 1))}
            G_sub = nx.relabel_nodes(G_sub, relabeling)
            G_sub.graph["label"] = data_graph_labels[idx]
            graphs.append(G_sub)
            node_num_list.append(G_sub.number_of_nodes())
        pylogger.info(f"number of graphs: {graph_num}")
        pylogger.info(f"average number of nodes: {sum(node_num_list) / len(node_num_list)}")

        if output_type == "pyg":
            for g in graphs:
                node_features = torch.stack(list(nx.get_node_attributes(g, "x").values()))
                gr = from_networkx(g)
                gr.x = node_features
                gr.y = g.graph["label"]
                graphs_list.append(gr)
                features_list.append(node_features)
        elif output_type == "networkx":
            for g in graphs:
                graphs_list.append(g)
                features_list.append(list(nx.get_node_attributes(g, "x").values()))
        else:
            raise Exception("output type must be pyg or networkx")
    return graphs_list, features_list


def write_TU_format(graph_list: List[nx.Graph], path: Path, dataset_name):
    data_list = to_data_list(graph_list)
    data_batch = Batch.from_data_list(data_list)
    pandas.DataFrame(
        data_batch.edge_index.T.numpy() + 1, index=list(range(data_batch.num_edges)), columns=["1", "2"]
    ).to_csv(path / (dataset_name + "_A.txt"), header=False, index=False)
    pandas.DataFrame(data_batch.batch.numpy() + 1, index=list(range(data_batch.num_nodes)), columns=[1]).to_csv(
        path / (dataset_name + "_graph_indicator.txt"), header=False, index=False
    )
    pandas.DataFrame(
        data_batch.x.numpy(), index=list(range(data_batch.num_nodes)), columns=list(range(data_batch.x.shape[-1]))
    ).to_csv(path / (dataset_name + "_node_attributes.txt"), header=False, index=False, float_format="%.6f")
    pandas.DataFrame(data_batch.y.numpy(), index=list(range(data_batch.num_graphs)), columns=[1]).to_csv(
        path / (dataset_name + "_graph_labels.txt"), header=False, index=False
    )


def to_data_list(graph_list) -> List[Data]:
    """
    Converts a list of Networkx graphs to a list of PyG Data objects
    :param graph_list: list of Networkx graphs

    :return:
    """
    data_list = []

    for G in graph_list:
        edge_index = get_edge_index_from_nx(G)

        x = [feat.tolist() for feat in nx.get_node_attributes(G, "x").values()]
        data = Data(
            edge_index=edge_index, num_nodes=G.number_of_nodes(), x=torch.tensor(x), y=torch.tensor(G.graph["label"])
        )

        data_list.append(data)

    return data_list


def get_edge_index_from_nx(G: nx.Graph) -> Tensor:
    """
    Extracts edge index from networkx graph
    :param G: networkx graph
    :return: tensor ~ (2, num_edges) containing all the edges in the graph G
    """
    # shape (num_edges*2, 2)
    edges_tensor = torch.tensor(list([(edge[0], edge[1]) for edge in G.edges]), dtype=torch.long)
    edges_tensor_reverse = torch.tensor(list([(edge[1], edge[0]) for edge in G.edges]), dtype=torch.long)

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
    split_seq_2 = sequence[support_upperbound + 1 :]

    return split_seq_1, split_seq_2
