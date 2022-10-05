import pickle as pkl
import sys

import networkx
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from hydra.utils import instantiate

from nn_core.common import PROJECT_ROOT
from src.discrete_diffusion.utils import adj_to_edge_index


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open(PROJECT_ROOT / "data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(PROJECT_ROOT / "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


class CoraGenerator:

    def __init__(self, num_samples: int = 3, graph_type='cora'):
        self.num_samples = num_samples

    def generate_data(self, save_data=None):
        """
        :param save_path:
        :return:
        """
        generated_graphs = []
        features_list = []
        for _ in range(self.num_samples):
            adj, features = load_data('cora')
            adj = torch.tensor(adj.todense())
            adj = adj_to_edge_index(adj)
            graph = networkx.convert.from_edgelist(zip(adj.tolist()[0], adj.tolist()[1]))
            x = features.todense().tolist()
            # networkx.set_node_attributes(graph, networkx.betweenness_centrality(graph), "x")
            networkx.set_node_attributes(graph, x, "x")
            generated_graphs.append(graph)
            features_list.append(x)

        # for i in range(self.num_samples):
        #     params = {}
        #
        #     if self.nx_params is not None:
        #         for k, v_list in self.nx_params.items():
        #             params[k] = np.random.choice(v_list)
        #
        #     graph = instantiate(self.nx_generator, **params)
        #     graph: nx.Graph = nx.relabel.convert_node_labels_to_integers(graph)
        #
        #     # for i in range(graph.number_of_nodes()):
        #     #     graph.nodes[i]["x"] = 1.0
        #     # graph.nodes[i]["x"] = one_hot(torch.tensor(i), graph.number_of_nodes()).float()
        #     nx.set_node_attributes(graph, nx.betweenness_centrality(graph), "x")
        #
        #     generated_graphs.append(graph)
        #     features_list.append([graph.nodes[i]["x"] for i in range(graph.number_of_nodes())])
        #
        #     if save_path:
        #         with open(f"{save_path}/{self.graph_type}_{i}.torch", "wb") as f:
        #             torch.save(obj=graph, f=f)

        return generated_graphs, features_list
