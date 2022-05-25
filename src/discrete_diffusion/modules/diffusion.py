import functools
from copy import copy
from typing import List

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.cm import coolwarm
from networkx.generators import erdos_renyi_graph
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from discrete_diffusion.utils import (
    adj_to_edge_index,
    edge_index_to_adj,
    get_data_from_edge_index,
    get_example_from_batch,
    get_graph_sizes_from_batch,
    unflatten_adj,
)


class Diffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: DictConfig,
        feature_dim: int,
        diffusion_speed: float,
        timesteps: int,
        threshold_sample: float,
    ):
        super().__init__()

        self.num_timesteps = timesteps

        assert diffusion_speed <= 0.5
        self.diffusion_speed = diffusion_speed
        self.threshold_sample = threshold_sample

        self.denoise_fn = instantiate(
            denoise_fn, node_embedder=denoise_fn.node_embedder, feature_dim=feature_dim, _recursive_=False
        )

        self.register_buffer("Qt", self.construct_transition_matrices())

    def construct_transition_matrices(self) -> torch.Tensor:
        """
        Constructs a tensor (T, 2, 2) containing for each timestep t in T the
        transition probabilities
        """
        Qts = []

        for t in range(self.num_timesteps + 1):
            flip_prob = 0.5 * (1 - (1 - 2 * self.diffusion_speed) ** t)
            not_flip_prob = 1 - flip_prob

            Qt = torch.tensor(
                [
                    [not_flip_prob, flip_prob],
                    [flip_prob, not_flip_prob],
                ],
            )
            Qts.append(Qt)

        Qts = torch.stack(Qts, dim=0)
        assert Qts.shape == (self.num_timesteps + 1, 2, 2)
        return Qts

    def forward(self, x_start: Batch, *args, **kwargs):

        random_timesteps = self.sample_timesteps(x_start)  # (B,)
        x_noisy = self.forward_diffusion(x_start, random_timesteps)
        # (all_possible_edges_batch, 2)
        q_noisy = self.backward_diffusion(
            x_start_batch=x_start, t_batch=random_timesteps, x_t_batch=x_noisy
        )  # (B * N_edges, 2)
        # (all_possible_edges_batch)
        q_approx = self.denoise_fn(x_noisy, random_timesteps)  # (B * N_edges,)
        loss = self.compute_loss(q_approx=q_approx, q_target=q_noisy[:, 1])
        return loss

    def sample_timesteps(self, x_start: Batch) -> torch.Tensor:
        """
        Sample a batch of random timesteps.

        :param x_start:

        :return:
        """
        batch_size = x_start.num_graphs
        random_timesteps = torch.randint(1, self.num_timesteps + 1, (batch_size,)).type_as(x_start["edge_index"])
        return random_timesteps

    def forward_diffusion(self, x_start: Batch, random_timesteps: torch.Tensor) -> Batch:
        """

        :param x_start:
        :param random_timesteps:

        :return:
        """
        batch_size = x_start.num_graphs

        # (B, 2, 2)
        Q_batch = self.Qt[random_timesteps]

        assert Q_batch.shape == (batch_size, 2, 2)

        adj = edge_index_to_adj(x_start.edge_index, x_start.num_nodes)
        adj_with_flip_probs = Q_batch[x_start.batch, adj, :]
        adj_noisy_unmasked = Categorical(adj_with_flip_probs).sample().type_as(adj)

        length_batches = x_start.ptr[1:] - x_start.ptr[:-1]
        mask = torch.block_diag(*[torch.triu(torch.ones(l, l), diagonal=1) for l in length_batches]).type_as(adj)

        adj_noisy_triu = adj_noisy_unmasked * mask
        adj_noisy = adj_noisy_triu + adj_noisy_triu.T
        x_noisy = copy(x_start)
        x_noisy.edge_index = adj_to_edge_index(adj_noisy)

        return x_noisy

    def backward_diffusion(self, x_start_batch: Batch, t_batch: torch.Tensor, x_t_batch: Batch) -> torch.Tensor:
        """
        Compute q(xt-1 | xt,x0) = (Q0_{:,xt} x Qt-1_{x0,:}) / Qt_{x0,xt}

        :param x_start_batch:
        :param t_batch:
        :param x_t_batch:

        :return: tensor (num_possible_edges_batch, 2)
        """
        batch_size = x_start_batch.num_graphs
        t_batch = t_batch.long()

        # (2, 2)
        Q_likelihood = self.Qt[1]
        # (B, 2, 2)
        Q_prior_batch = self.Qt[t_batch - 1]
        Q_evidence_batch = self.Qt[t_batch]

        assert Q_likelihood.shape == (2, 2)
        assert Q_prior_batch.shape == Q_evidence_batch.shape == (batch_size, 2, 2)

        adj_x_t = edge_index_to_adj(x_t_batch.edge_index, x_t_batch.num_nodes)
        adj_x_start = edge_index_to_adj(x_start_batch.edge_index, x_start_batch.num_nodes)

        likelihood = Q_likelihood[:, adj_x_t].permute(1, 2, 0)

        prior = Q_prior_batch[x_start_batch.batch, adj_x_start, :]

        evidence = Q_evidence_batch[x_start_batch.batch, adj_x_start, adj_x_t, None]

        q_backward = (likelihood * prior) / evidence

        length_batches = x_start_batch.ptr[1:] - x_start_batch.ptr[:-1]
        mask = torch.block_diag(*[torch.triu(torch.ones(l, l), diagonal=1) for l in length_batches]).type(torch.bool)

        q_backward_all = q_backward[mask]

        return q_backward_all

    @staticmethod
    def compute_loss(
        q_target: torch.Tensor,
        q_approx: torch.Tensor,
    ):
        cross_ent = F.binary_cross_entropy_with_logits(q_approx, q_target, reduction="mean")
        ent = F.binary_cross_entropy(q_target, q_target, reduction="mean")
        loss = cross_ent - ent
        # loss = F.mse_loss(q_approx, q_target, reduce="mean")
        return loss

    @torch.no_grad()
    def sampling_step(self, x_noisy: Batch, t: torch.Tensor) -> (Batch, torch.Tensor):

        batch_size = x_noisy.num_graphs

        # flattened concatenation of edge probabilities in the batch
        # (all_possible_edges_in_batch, )
        edge_similarities = self.denoise_fn(x_noisy, t)
        edge_probs = self.compute_edge_probs(edge_similarities)
        # edge_probs = edge_similarities

        edge_probs_expanded = torch.stack((1 - edge_probs, edge_probs), dim=-1)

        if t[0] != 1:
            # (all_possible_edges_in_batch, )
            flattened_sampled_adjs = Categorical(probs=edge_probs_expanded).sample().long()
        else:
            flattened_sampled_adjs = (edge_probs > self.threshold_sample).long()

        assert flattened_sampled_adjs.shape == edge_probs.shape

        graph_sizes = get_graph_sizes_from_batch(x_noisy)
        flat_idx = x_noisy.ptr[0]
        flattened_adj_indices = [flat_idx]
        for i in range(batch_size):
            flat_idx = flat_idx + torch.div(graph_sizes[i] * (graph_sizes[i] - 1), 2, rounding_mode="floor")
            flattened_adj_indices.append(flat_idx)

        graph_list: List[Data] = []
        edge_probs_list: List[torch.Tensor] = []

        for i in range(len(graph_sizes)):

            num_nodes = graph_sizes[i]

            # (n*n)
            flattened_adj_g = flattened_sampled_adjs[flattened_adj_indices[i] : flattened_adj_indices[i + 1]]
            # (n, n)
            adj = unflatten_adj(flattened_adj_g, num_nodes)

            flattened_edge_probs_g = edge_probs[flattened_adj_indices[i] : flattened_adj_indices[i + 1]]

            edge_probs_g = unflatten_adj(flattened_edge_probs_g, num_nodes)

            edge_index = adj_to_edge_index(adj)
            node_features = x_noisy.x[x_noisy.ptr[i] : x_noisy.ptr[i + 1]]
            graph = get_data_from_edge_index(edge_index, node_features)

            graph_list.append(graph)
            edge_probs_list.append(edge_probs_g)

        graph_batch = Batch.from_data_list(graph_list)

        return graph_batch, edge_probs_list

    def compute_edge_probs(self, edge_similarities):
        return torch.sigmoid(edge_similarities)

    @torch.no_grad()
    def sample(self, features_list, num_nodes_samples) -> (Batch, plt.Figure):
        """
        Generate graphs

        :return:
        """

        generated_graphs: List[Data] = []

        for num_nodes in num_nodes_samples:

            graph = self.generate_noisy_graph(num_nodes)

            idx = torch.randint(len(features_list) - 1, (1,)).item()
            while len(features_list[idx]) != num_nodes:
                idx = torch.randint(len(features_list) - 1, (1,)).item()
            graph.x = features_list[idx].type_as(self.Qt)

            generated_graphs.append(graph)

        graphs_batch = Batch.from_data_list(generated_graphs)

        graphs_batch, fig_adj = self.sample_and_plot(graphs_batch)

        return graphs_batch.to_data_list(), fig_adj

    def generate_noisy_graph(self, num_nodes):
        nx_graph = erdos_renyi_graph(n=num_nodes, p=0.5)
        graph = from_networkx(nx_graph)
        graph.edge_index = graph.edge_index.type_as(self.Qt[0]).long()
        return graph

    def sample_and_plot(self, graphs_batch):
        num_generated_graphs = graphs_batch.num_graphs

        side = 4
        freq = self.num_timesteps // (side**2 - 1) + 1

        fig_adj, axs_adj = plt.subplots(side, side, constrained_layout=True)
        axs_adj = axs_adj.flatten()

        for i in tqdm(
            reversed(range(1, self.num_timesteps + 1)), desc="Sampling loop time step", total=self.num_timesteps + 1
        ):

            times = torch.full((num_generated_graphs,), i).type_as(self.Qt[0])

            if i == self.num_timesteps:
                graph = get_example_from_batch(graphs_batch, 0)
                adj = edge_index_to_adj(graph.edge_index, graph.num_nodes)
                axs_adj[0].imshow(adj.cpu().detach(), cmap=coolwarm, vmin=0, vmax=1)
                axs_adj[0].set_title("initial noise")

            graphs_batch, edge_probs_list = self.sampling_step(graphs_batch, times)

            if (i % freq == 0) or (i == 1):
                j = i // freq
                axs_adj[side**2 - 1 - j].imshow(edge_probs_list[0].cpu().detach(), cmap=coolwarm, vmin=0, vmax=1)
                axs_adj[side**2 - 1 - j].set_title("noise = " + str(i))

        fig_adj.colorbar(cm.ScalarMappable(cmap=coolwarm))

        return graphs_batch, fig_adj


class GroundTruthDiffusion(Diffusion):
    def __init__(
        self,
        denoise_fn: DictConfig,
        feature_dim: int,
        diffusion_speed: float,
        timesteps: int,
        threshold_sample: float,
        ref_graph_edges,
        ref_graph_feat,
    ):
        super(Diffusion, self).__init__()

        self.num_timesteps = timesteps
        self.diffusion_speed = diffusion_speed
        self.threshold_sample = threshold_sample
        self.register_buffer("ref_graph_edges", ref_graph_edges)
        self.register_buffer("ref_graph_feat", ref_graph_feat)
        Qt = self.construct_transition_matrices()
        self.register_buffer("Qt", Qt)

        self.denoise_fn = instantiate(
            denoise_fn, ref_graph_edges=ref_graph_edges, ref_graph_feat=ref_graph_feat, Qt=Qt, _recursive_=False
        )

        self.dummy_par = nn.Linear(1, 1)

    def forward(self, x_start: Batch, *args, **kwargs):
        dummy_loss = F.mse_loss(self.dummy_par(torch.ones((1,)).type_as(self.Qt)), torch.zeros((1,)).type_as(self.Qt))
        return dummy_loss

    # def generate_noisy_graph(self, num_nodes) -> Data:
    #     ref_graph = get_data_from_edge_index(self.ref_graph_edges, self.ref_graph_feat)
    #     ref_batch = Batch.from_data_list([ref_graph])
    #     t = torch.tensor([self.num_timesteps]).type_as(ref_graph.edge_index)
    #     noisy_batch = self.forward_diffusion(ref_batch, t)
    #     noisy_graph = get_data_from_edge_index(noisy_batch.edge_index, noisy_batch.x)
    #     return noisy_graph
