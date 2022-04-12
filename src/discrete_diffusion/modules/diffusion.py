# import matplotlib.pyplot as plt
from typing import List

import networkx.generators
import torch
import torch.nn.functional as F
import torch_geometric.utils
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from discrete_diffusion.utils import adj_to_edge_index, edge_index_to_adj


class Diffusion(nn.Module):
    def __init__(self, denoise_fn: DictConfig, feature_dim, diffusion_speed, timesteps, num_nodes_samples: List):
        super().__init__()
        self.denoise_fn = instantiate(
            denoise_fn, node_embedder=denoise_fn.node_embedder, feature_dim=feature_dim, _recursive_=False
        )
        self.num_timesteps = int(timesteps)
        self.diffusion_speed = diffusion_speed
        self.register_buffer("Qt", self.get_Qt())
        self.num_nodes_samples = num_nodes_samples

    def forward(self, x_start: Batch, *args, **kwargs):
        batch_size = x_start.ptr.shape[0] - 1
        t = torch.randint(0, self.num_timesteps, (batch_size,)).type_as(x_start["edge_index"])

        x_noisy = self.forward_diffusion(x_start, t)
        q_noisy = self.backward_diffusion(
            x_start_batch=x_start, t_batch=t, x_t_batch=x_noisy
        )  # (all_possible_edges_batch, 2)
        q_approx = self.denoise_fn(x_noisy, t)  # (all_possible_edges_batch, 2)

        loss = self.loss(q_noisy, q_approx)

        return loss

    def get_Qt(self):
        Qt = torch.empty(self.num_timesteps, 2, 2)
        for t in range(1, self.num_timesteps + 1):
            flip_prob = 0.5 * (1 - (1 - 2 * self.diffusion_speed) ** t)
            not_flip_prob = 1 - flip_prob
            Q = torch.tensor(
                [
                    [not_flip_prob, flip_prob],
                    [flip_prob, not_flip_prob],
                ],
            )
            Qt[t - 1] = Q
        return Qt

    @torch.no_grad()
    def sampling_step(self, x: Batch, t) -> Batch:

        # Returns the flattened concatenation of adj matrices in the batch
        logits = self.denoise_fn(x, t)
        sample = torch.distributions.Categorical(logits=logits).sample()
        sample = sample.type(torch.float)

        # Build a Batch from it
        num_nodes_batches = x.ptr[1:] - x.ptr[:-1]
        graphs_list = []

        for i in range(len(num_nodes_batches)):
            num_nodes = num_nodes_batches[i]
            adj = sample[x.ptr[i] ** 2 : x.ptr[i + 1] ** 2].reshape((num_nodes, num_nodes))
            edge_list, _ = torch_geometric.utils.dense_to_sparse(adj)
            graph = Data(x=torch.ones(num_nodes).type_as(self.Qt[0]), edge_index=edge_list.type_as(self.Qt[0]).long())
            graphs_list.append(graph)

        graph_batch = Batch.from_data_list(graphs_list)

        return graph_batch

    @torch.no_grad()
    def sample(self):

        b = len(self.num_nodes_samples)

        generated_graphs: List[Data] = []
        for n in self.num_nodes_samples:
            nx_graph = networkx.generators.erdos_renyi_graph(n=n, p=0.5)
            graph = from_networkx(nx_graph)
            graph.edge_index = graph.edge_index.type_as(self.Qt[0]).long()
            graph.x = torch.ones(n).type_as(self.Qt[0])
            generated_graphs.append(graph)

        graphs_batch = Batch.from_data_list(generated_graphs)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            times = torch.full((b,), i).type_as(self.Qt[0])
            graphs_batch = self.sampling_step(graphs_batch, times)

        return graphs_batch.to_data_list()

    def backward_diffusion(self, x_start_batch: Batch, t_batch: torch.Tensor, x_t_batch: Batch) -> torch.Tensor:
        Qt = self.Qt

        # Expression for q(xt-1 | xt,x0) = (Q0_{:,xt} x Qt-1_{x0,:}) / Qt_{x0,xt}
        Q_likelihood = Qt[0]  # [b, num_cat, num_cat]
        Q_prior_batch = Qt[t_batch - 1]
        Q_evidence_batch = Qt[t_batch]

        x_start_data_list: List[Data] = x_start_batch.to_data_list()
        x_t_data_list: List[Data] = x_t_batch.to_data_list()
        batch_size = t_batch.shape[0]

        q_backward_list = torch.tensor([]).type_as(Q_likelihood)
        for b in range(batch_size):

            x_start, x_t, Q_prior, Q_evidence = (
                x_start_data_list[b],
                x_t_data_list[b],
                Q_prior_batch[b],
                Q_evidence_batch[b],
            )

            # (n, n)
            adj_x_t = edge_index_to_adj(x_t.edge_index, x_t.num_nodes)
            # (n, n)
            adj_x_start = edge_index_to_adj(x_start.edge_index, x_start.num_nodes)

            # (n, n, 2)
            likelihood = Q_likelihood[adj_x_t]
            # (n, n, 2)
            prior = Q_prior[adj_x_start]
            # (n, n)
            evidence = Q_evidence[adj_x_start, adj_x_t]

            # (n, n, 2)
            q_backward = (likelihood * prior) / evidence.unsqueeze(-1)

            # (n*n, 2)
            q_backward = q_backward.flatten(0, 1)

            q_backward_list = torch.cat((q_backward_list, q_backward))
            # q_backward_list.append(q_backward)

        return q_backward_list

    def loss(
        self,
        q_noisy,
        q_recon,
    ):
        # TODO: find out what's going on
        # q_recon = torch.softmax(q_recon, dim=-1)
        # q_noisy = torch.argmax(torch.softmax(q_noisy, dim=-1), dim=-1)
        loss = F.cross_entropy(q_recon, q_noisy, reduction="sum")

        return loss

    def forward_diffusion(self, x_start: Batch, t: torch.Tensor):
        """

        :param x_start:
        :param t:
        :return:
        """
        Q_batch = self.Qt[t]  # [b, n, n]

        data_list: List[Data] = x_start.to_data_list()

        for b, data in enumerate(data_list):

            adj = edge_index_to_adj(data.edge_index, data.num_nodes)
            Q = Q_batch[b]

            q = Q[adj]
            x_noisy = torch.distributions.Categorical(q).sample().type_as(adj)  # [n, n]

            new_edge_index = adj_to_edge_index(x_noisy)
            data.edge_index = new_edge_index

        x_noisy = Batch.from_data_list(data_list)

        return x_noisy
