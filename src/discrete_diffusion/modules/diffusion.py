# import matplotlib.pyplot as plt
import math
from typing import List

import networkx as nx
import networkx.generators
import torch
import torch.nn.functional as F
import torch_geometric.utils
import wandb
from hydra.utils import instantiate
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.cm import coolwarm
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

        # TODO: LINEA PER DEBUGGARE DIO ME NE SCAMPI NON LASCIARLA
        # t = t * 0 + 1

        x_noisy = self.forward_diffusion(x_start, t)
        q_noisy = self.backward_diffusion(
            x_start_batch=x_start, t_batch=t, x_t_batch=x_noisy
        )  # (all_possible_edges_batch, 2)
        q_approx = self.denoise_fn(x_noisy, t)  # (all_possible_edges_batch, 2)

        loss = self.loss(q_target=q_noisy[:, 0], q_approx=q_approx)

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
    def sampling_step(self, x: Batch, t) -> (Batch, torch.Tensor):

        # Returns the flattened concatenation of adj matrices in the batch
        probs = self.denoise_fn(x, t)
        probs_expanded = torch.stack((probs, 1 - probs), dim=-1)
        sample = torch.distributions.Categorical(probs=probs_expanded).sample()
        sample = sample.type(torch.float)

        # Build a Batch from it
        num_nodes_batches = x.ptr[1:] - x.ptr[:-1]
        graphs_list = []
        adj_list = []

        for i in range(len(num_nodes_batches)):
            num_nodes = num_nodes_batches[i]
            adj = sample[x.ptr[i] ** 2 : x.ptr[i + 1] ** 2].reshape((num_nodes, num_nodes))
            adj_soft = probs[x.ptr[i] ** 2 : x.ptr[i + 1] ** 2].reshape((num_nodes, num_nodes))
            edge_list, _ = torch_geometric.utils.dense_to_sparse(adj)
            graph = Data(x=torch.ones(num_nodes).type_as(self.Qt[0]), edge_index=edge_list.type_as(self.Qt[0]).long())
            graphs_list.append(graph)
            adj_list.append(adj_soft)

        graph_batch = Batch.from_data_list(graphs_list)

        return graph_batch, adj_list

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

        side = math.sqrt(self.num_timesteps // 10)
        batch_size_h = math.ceil(side)
        batch_size_w = math.ceil(side)
        # fig_graphs, axs_graphs = plt.subplots(batch_size_h, batch_size_w)
        fig_adj, axs_adj = plt.subplots(batch_size_h, batch_size_w)
        # axs_graphs = axs_graphs.flatten()
        axs_adj = axs_adj.flatten()

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            times = torch.full((b,), i).type_as(self.Qt[0])
            graphs_batch, adj_list = self.sampling_step(graphs_batch, times)
            if i % 10 == 0:
                # G = graphs_batch.to_data_list()[0]
                # G_nx = torch_geometric.utils.to_networkx(G)
                # nx.draw(G_nx, with_labels=True, ax=axs_graphs[i // 10], node_size=1)
                axs_adj[i // 10].imshow(adj_list[0].cpu().detach(), cmap=coolwarm, vmin=0, vmax=1)
        fig_adj.colorbar(cm.ScalarMappable(cmap=coolwarm))

        return graphs_batch.to_data_list(), fig_adj

    def backward_diffusion(self, x_start_batch: Batch, t_batch: torch.Tensor, x_t_batch: Batch) -> torch.Tensor:
        Qt = self.Qt

        # Expression for q(xt-1 | xt,x0) = (Q0_{:,xt} x Qt-1_{x0,:}) / Qt_{x0,xt}
        Q_likelihood = Qt[0]  # [b, num_cat, num_cat]
        Q_prior_batch = Qt[t_batch - 1]
        Q_evidence_batch = Qt[t_batch]

        x_start_data_list: List[Data] = x_start_batch.to_data_list()
        x_t_data_list: List[Data] = x_t_batch.to_data_list()
        batch_size = t_batch.shape[0]

        q_backward_all = torch.tensor([]).type_as(Q_likelihood)
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

            q_backward_all = torch.cat((q_backward_all, q_backward))
            # q_backward_list.append(q_backward)

        return q_backward_all

    def loss(
        self,
        q_target,
        q_approx,
    ):
        # TODO: find out what's going on
        # q_recon = (q_recon - torch.min(q_recon)) / (torch.max(q_recon) - torch.min(q_recon))
        # q_noisy = torch.argmax(torch.softmax(q_noisy, dim=-1), dim=-1)
        loss = F.binary_cross_entropy(q_approx, q_target, reduction="mean")

        # - (q_noisy[0, 0] * torch.log2(q_recon[0]) + (1 - q_noisy[0, 0]) * torch.log2(1 - q_recon[0]))

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
