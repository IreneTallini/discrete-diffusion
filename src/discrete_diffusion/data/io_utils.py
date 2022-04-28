import math
import os
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Data


class Node:
    def __init__(self, tag, neighbors, attrs):
        self.tag = tag
        self.neighbors = neighbors
        self.attrs = attrs


def load_data(dir_path, dataset_name, feature_params):
    """
    Loads a TU graph dataset.

    :param dir_path: path to the directory containing the dataset
    :param dataset_name: name of the dataset
    :param feature_params: params regarding data features
    :return:
    """

    graph_list = load_graph_list(dir_path, dataset_name)

    class_to_label_dict = get_classes_to_label_dict(graph_list)
    data_list = to_data_list(graph_list, class_to_label_dict, feature_params)

    set_node_features(
        data_list,
        feature_params=feature_params,
    )

    return data_list, class_to_label_dict


def load_graph_list(dir_path, dataset_name):
    """
    Loads a graph dataset as a list of networkx graphs

    :param dir_path: path to the directory containing the dataset
    :param dataset_name: name of the dataset

    :return: graph_list: list of networkx graphs
    """
    dataset_path = f"{os.path.join(dir_path, dataset_name)}.txt"

    graph_list = []
    with open(dataset_path, "r") as f:
        num_graphs = int(f.readline().strip())

        for graph_ind in range(num_graphs):

            graph: nx.Graph = parse_graph(f)
            graph_list.append(graph)

    return graph_list


def to_data_list(graph_list, class_to_label_dict, feature_params) -> List[Data]:
    """
    Converts a list of Networkx graphs to a list of PyG Data objects

    :param graph_list: list of Networkx graphs
    :param class_to_label_dict: mapping original class to integer label

    :return:
    """
    data_list = []

    for G in graph_list:
        edge_index = get_edge_index_from_nx(G)
        label = torch.tensor(class_to_label_dict[G.graph["class"]], dtype=torch.long).unsqueeze(0)

        data = Data(
            edge_index=edge_index,
            num_nodes=G.number_of_nodes(),
            y=label,
            degrees=get_degree_tensor_from_nx(G),
            tags=get_tag_tensor_from_nx(G),
            num_cycles=get_num_cycles_from_nx(G, feature_params["max_considered_cycle_len"]),
        )

        data_list.append(data)

    return data_list


def parse_graph(file_descriptor):
    """
    Parses a single graph from file

    :param file_descriptor: file formatted accordingly to TU datasets

    :return: networkx graph
    """

    graph_header = file_descriptor.readline().strip().split()
    num_nodes, cls = [int(w) for w in graph_header]

    G = nx.Graph()
    G.graph["class"] = str(cls)

    for node_ind in range(num_nodes):

        node: Node = parse_node(file_descriptor)

        G.add_node(node_ind, tag=node.tag, attrs=node.attrs)

        for neighbor in node.neighbors:
            G.add_edge(node_ind, neighbor)
            G.add_edge(neighbor, node_ind)

    assert len(G) == num_nodes

    return G


def parse_node(file_descriptor):
    """
    Parses a single node from file, corresponding to a row having format
        tag num_neighbors nghbr_1 nghbr_2 ... attr_1 attr_2 ...

    :param file_descriptor: file formatted accordingly to TU datasets
    :return: Node with tag, neighbors list and possibly attributes
    """

    node_row = file_descriptor.readline().strip().split()

    node_header = node_row[0:2]
    tag, num_neighbors = int(node_header[0]), int(node_header[1])

    # attributes come after the header (tag and num_neighbors) and all the neighbors
    attr_starting_index = 2 + num_neighbors

    neighbors = [int(w) for w in node_row[2:attr_starting_index]]

    attrs = [float(w) for w in node_row[attr_starting_index:]]
    attrs = np.array(attrs) if attrs else None

    return Node(tag, neighbors, attrs)


def get_degree_tensor_from_nx(G: nx.Graph) -> Tensor:
    """
    Returns node degrees as a tensor
    :param G: networkx graph

    :return: tensor ~ (num_nodes) with tensor[i] = degree of node i
    """
    degree_list = sorted(list(G.degree), key=lambda x: x[0])

    return torch.tensor([pair[1] for pair in degree_list])


def get_tag_tensor_from_nx(G: nx.Graph) -> Tensor:
    """
    Returns node tags as a tensor
    :param G: networkx graph

    :return: tensor ~ (num_nodes) with tensor[i] = tag of node i
    """

    tag_dict = nx.get_node_attributes(G, "tag")
    tag_tuples = [(key, value) for key, value in tag_dict.items()]

    node_and_tags_sorted_by_node = sorted(tag_tuples, key=lambda t: t[0])
    tags_sorted_by_node = [tup[1] for tup in node_and_tags_sorted_by_node]

    return torch.tensor(tags_sorted_by_node)


def get_num_cycles_from_nx(G: nx.Graph, max_considered_cycle_len) -> Tensor:

    A = torch.Tensor(nx.adjacency_matrix(G).todense())
    A_k = torch.clone(A)

    num_cycles = []
    for k in range(max_considered_cycle_len):
        A_k = A_k.t() @ A
        num_cycles_len_k = A_k.diagonal()

        num_cycles.append(torch.tensor(num_cycles_len_k))

    return torch.stack(num_cycles, dim=0)


def set_node_features(data_list: List[Data], feature_params: Dict):
    """
    Adds to each data in data_list either the tags, the degrees or both as node features
    In place function

    :param data_list: list of preprocessed graphs as PyG Data objects
    :param feature_params:

    """

    # contains for each graph G its node features, where each feature is a vector of length N_G
    all_node_features = []

    if "tag" in feature_params["features_to_consider"]:
        all_tags = torch.cat([data.tags for data in data_list], 0)
        one_hot_tags = get_one_hot_attrs(all_tags, data_list)
        all_node_features = initialize_or_concatenate(all_node_features, one_hot_tags)

    if "degree" in feature_params["features_to_consider"]:
        all_degrees = torch.cat([data.degrees for data in data_list], 0)
        one_hot_degrees = get_one_hot_attrs(all_degrees, data_list)
        all_node_features = initialize_or_concatenate(all_node_features, one_hot_degrees)

    if "num_cycles" in feature_params["features_to_consider"]:
        for k in range(1, feature_params["max_considered_cycle_len"]):
            num_cycles = [data.num_cycles[k] for data in data_list]
            all_node_features = initialize_or_concatenate(all_node_features, num_cycles)

    for data, node_features in zip(data_list, all_node_features):

        assert data.num_nodes == node_features.shape[0]
        data["x"] = node_features
        data["num_sample_edges"] = data.edge_index.shape[1]
        data["degrees"] = None
        data["tags"] = None
        data["num_cycles"] = None


def initialize_or_concatenate(all_node_features, feature_to_add):

    if len(all_node_features) == 0:
        return feature_to_add

    num_graphs = len(all_node_features)

    new_all_node_features = [torch.cat((all_node_features[i], feature_to_add[i]), dim=1) for i in range(num_graphs)]

    return new_all_node_features


def get_one_hot_attrs(attrs, data_list):
    """

    :param attrs:
    :param data_list:
    :return:
    """
    # unique_attrs contains the unique values found in attrs,
    # corrs contains the indices of the unique array that reconstruct the input array
    unique_attrs, corrs = np.unique(attrs, return_inverse=True, axis=0)
    num_different_attrs = len(unique_attrs)

    # encode
    all_one_hot_attrs = []
    pointer = 0

    for data in data_list:
        hots = torch.LongTensor(corrs[pointer : pointer + data.num_nodes])
        data_one_hot_attrs = F.one_hot(hots, num_different_attrs).float()

        all_one_hot_attrs.append(data_one_hot_attrs)
        pointer += data.num_nodes

    return all_one_hot_attrs


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


def get_classes_to_label_dict(graph_list) -> Dict:
    """
    Obtains all the classes present in the data and maps them to progressive integers.

    :param graph_list: list of networkx graphs

    :return: map that maps each string class to an integer
    """

    all_classes = {graph.graph["class"] for graph in graph_list}
    all_classes_sorted = sorted([int(cls) for cls in all_classes])
    class_to_label_dict = {str(cls): label for label, cls in enumerate(all_classes_sorted)}

    return class_to_label_dict


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
