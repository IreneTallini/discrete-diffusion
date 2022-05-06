from random import randint
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

from discrete_diffusion.utils import adj_to_edge_index, edge_index_to_adj, get_graph_sizes_from_batch


class Diffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: DictConfig,
        feature_dim: int,
        diffusion_speed: float,
        timesteps: int,
        num_nodes_samples: List[int],
    ):
        super().__init__()

        self.num_timesteps = timesteps
        self.diffusion_speed = diffusion_speed
        self.num_nodes_samples = num_nodes_samples

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

        for t in range(1, self.num_timesteps + 1):

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
        assert Qts.shape == (self.num_timesteps, 2, 2)
        return Qts

    def forward(self, x_start: Batch, *args, **kwargs):

        random_timesteps = self.sample_timesteps(x_start)

        x_noisy = self.forward_diffusion(x_start, random_timesteps)

        # (all_possible_edges_batch, 2)
        q_noisy = self.backward_diffusion(x_start_batch=x_start, t_batch=random_timesteps, x_t_batch=x_noisy)

        # (all_possible_edges_batch)
        q_approx = self.denoise_fn(x_noisy, random_timesteps)

        loss = self.compute_loss(q_target=q_noisy[:, 1], q_approx=q_approx)

        return loss

    def sample_timesteps(self, x_start: Batch) -> torch.Tensor:
        """
        Sample a batch of random timesteps.

        :param x_start:

        :return:
        """
        batch_size = x_start.num_graphs

        random_timesteps = torch.randint(0, self.num_timesteps, (batch_size,)).type_as(x_start["edge_index"])

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

        data_list: List[Data] = x_start.to_data_list()
        noisy_data_list: List[Data] = []

        for b, data in enumerate(data_list):
            transition_matrix = Q_batch[b]
            noisy_data = self.compute_noisy_data(data, transition_matrix)
            noisy_data_list.append(noisy_data)

        x_noisy = Batch.from_data_list(noisy_data_list)

        assert Q_batch.shape == (batch_size, 2, 2)

        return x_noisy

    def compute_noisy_data(self, data: Data, transition_matrix: torch.Tensor) -> Data:
        """
        Compute a noisy adjacency matrix in accordance to the forward process

        :param data:
        :param transition_matrix: (2, 2) matrix of transition probabilities
        """
        adj = edge_index_to_adj(data.edge_index, data.num_nodes)

        # (n, n, 2)
        adj_with_flip_probs = transition_matrix[adj]

        # (n, n)
        x_noisy = Categorical(adj_with_flip_probs).sample().type_as(adj)

        new_edge_index = adj_to_edge_index(x_noisy)

        noisy_data = Data(edge_index=new_edge_index, num_nodes=data.num_nodes, x=data.x)

        assert adj_with_flip_probs.shape == (data.num_nodes, data.num_nodes, 2)
        assert x_noisy.shape == (data.num_nodes, data.num_nodes)

        return noisy_data

    def backward_diffusion(self, x_start_batch: Batch, t_batch: torch.Tensor, x_t_batch: Batch) -> torch.Tensor:
        """
        Compute q(xt-1 | xt,x0) = (Q0_{:,xt} x Qt-1_{x0,:}) / Qt_{x0,xt}

        :param x_start_batch:
        :param t_batch:
        :param x_t_batch:

        :return: tensor (num_possible_edges_batch, 2)
        """
        batch_size = x_start_batch.num_graphs

        # (2, 2)
        Q_likelihood = self.Qt[0]
        # (B, 2, 2)
        Q_prior_batch = self.Qt[t_batch - 1]
        Q_evidence_batch = self.Qt[t_batch]

        assert Q_likelihood.shape == (2, 2)
        assert Q_prior_batch.shape == Q_evidence_batch.shape == (batch_size, 2, 2)

        x_start_data_list: List[Data] = x_start_batch.to_data_list()
        x_t_data_list: List[Data] = x_t_batch.to_data_list()

        q_backward_list = []
        for x_start, x_t, Q_prior, Q_evidence in zip(x_start_data_list, x_t_data_list, Q_prior_batch, Q_evidence_batch):

            q_backward = self.compute_q_backward(Q_evidence, Q_likelihood, Q_prior, x_start, x_t)

            q_backward_list.append(q_backward)

        q_backward_all = torch.cat(q_backward_list, dim=0)

        return q_backward_all

    def compute_q_backward(
        self, Q_evidence: torch.Tensor, Q_likelihood: torch.Tensor, Q_prior: torch.Tensor, x_start: Data, x_t: Data
    ) -> torch.Tensor:
        """
        :param Q_evidence:
        :param Q_likelihood:
        :param Q_prior:
        :param x_start:
        :param x_t:

        :return:
        """
        assert x_t.num_nodes == x_start.num_nodes
        num_nodes = x_t.num_nodes

        # (n, n)
        adj_x_t = edge_index_to_adj(x_t.edge_index, num_nodes)
        # (n, n)
        adj_x_start = edge_index_to_adj(x_start.edge_index, num_nodes)

        # (n, n, 2)
        likelihood = Q_likelihood[adj_x_t]
        # (n, n, 2)
        prior = Q_prior[adj_x_start]
        # (n, n)
        evidence = Q_evidence[adj_x_start, adj_x_t]

        # (n, n, 2)
        q_backward = (likelihood * prior) / evidence.unsqueeze(-1)
        tril_indices = torch.tril_indices(num_nodes, num_nodes, offset=-1)

        # (n*(n-1)/2, 2)
        q_backward = q_backward[tril_indices[0], tril_indices[1], :]

        # q_backward = q_backward.flatten(0, 1)

        assert q_backward.shape == (num_nodes * (num_nodes - 1) / 2, 2)

        return q_backward

    def compute_loss(
        self,
        q_target: torch.Tensor,
        q_approx: torch.Tensor,
    ):
        loss = F.binary_cross_entropy_with_logits(q_approx, q_target, reduction="mean")

        return loss

    @torch.no_grad()
    def sampling_step(self, x_noisy: Batch, t: torch.Tensor) -> (Batch, torch.Tensor):

        batch_size = x_noisy.num_graphs

        # flattened concatenation of edge probabilities in the batch
        # (all_possible_edges_in_batch, )
        edge_similarities = self.denoise_fn(x_noisy, t)
        edge_probs = torch.sigmoid(edge_similarities)

        # edge_probs_expanded = torch.stack((edge_probs, 1 - edge_probs), dim=-1)
        edge_probs_expanded = torch.stack((1 - edge_probs, edge_probs), dim=-1)

        if t[0] != 0:
            # (all_possible_edges_in_batch, )
            flattened_sampled_adjs = Categorical(probs=edge_probs_expanded).sample().long()
        else:
            flattened_sampled_adjs = (edge_probs > 0.5).long()

        assert flattened_sampled_adjs.shape == edge_probs.shape

        graph_sizes = get_graph_sizes_from_batch(x_noisy)
        flat_idx = x_noisy.ptr[0]
        flattened_adj_indices = [flat_idx]
        for i in range(batch_size):
            # flat_idx = flat_idx + graph_sizes[i] ** 2
            flat_idx = flat_idx + graph_sizes[i] * (graph_sizes[i] - 1) // 2
            flattened_adj_indices.append(flat_idx)

        graphs_list: List[Data] = []
        edge_probs_list: List[torch.Tensor] = []

        for i in range(len(graph_sizes)):

            num_nodes = graph_sizes[i]

            # (n*n)
            flattened_adj_g = flattened_sampled_adjs[flattened_adj_indices[i] : flattened_adj_indices[i + 1]]
            # (n, n)
            adj = torch.zeros((num_nodes, num_nodes)).type_as(flattened_adj_g)
            tril_indices = torch.tril_indices(num_nodes, num_nodes, offset=-1)
            adj[tril_indices[0], tril_indices[1]] = flattened_adj_g
            adj = adj + adj.T
            # adj = flattened_adj_g.reshape((num_nodes, num_nodes))

            flattened_edge_probs_g = edge_probs[flattened_adj_indices[i] : flattened_adj_indices[i + 1]]
            edge_probs_g = torch.zeros((num_nodes, num_nodes)).type_as(flattened_edge_probs_g)
            edge_probs_g[tril_indices[0], tril_indices[1]] = flattened_edge_probs_g
            edge_probs_g = edge_probs_g + edge_probs_g.T
            # edge_probs_g = flattened_edge_probs_g.reshape((num_nodes, num_nodes))

            edge_index = adj_to_edge_index(adj)
            node_features = x_noisy.x[x_noisy.ptr[i] : x_noisy.ptr[i + 1]]
            graph = Data(x=node_features, edge_index=edge_index, num_nodes=num_nodes)

            graphs_list.append(graph)
            edge_probs_list.append(edge_probs_g)

        graph_batch = Batch.from_data_list(graphs_list)

        return graph_batch, edge_probs_list

    @torch.no_grad()
    def sample(self, train_data):
        """
        Generate graphs

        :return:
        """

        generated_graphs: List[Data] = []

        for sample_num_nodes in self.num_nodes_samples:

            nx_graph = erdos_renyi_graph(n=sample_num_nodes, p=0.5)
            data = from_networkx(nx_graph)
            data.edge_index = data.edge_index.type_as(self.Qt[0]).long()

            idx = torch.randint(len(train_data) - 1, (1,)).item()
            while train_data[idx].num_nodes != sample_num_nodes:
                idx = torch.randint(len(train_data) - 1, (1,)).item()
            data.x = train_data[idx].x

            generated_graphs.append(data)

        graphs_batch = Batch.from_data_list(generated_graphs)

        graphs_batch, fig_adj = self.sample_and_plot(graphs_batch)

        return graphs_batch.to_data_list(), fig_adj

    def sample_and_plot(self, graphs_batch):
        num_generated_graphs = len(self.num_nodes_samples)

        side = 4
        freq = self.num_timesteps // side**2 + 1

        fig_adj, axs_adj = plt.subplots(side, side)
        axs_adj = axs_adj.flatten()

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="Sampling loop time step", total=self.num_timesteps):

            times = torch.full((num_generated_graphs,), i).type_as(self.Qt[0])

            graphs_batch, edge_probs_list = self.sampling_step(graphs_batch, times)

            if i % freq == 0 or i == self.num_timesteps - 1:
                axs_adj[i // freq].imshow(edge_probs_list[0].cpu().detach(), cmap=coolwarm, vmin=0, vmax=1)
                axs_adj[i // freq].set_title("noise = " + str(i))

        fig_adj.colorbar(cm.ScalarMappable(cmap=coolwarm))

        return graphs_batch, fig_adj
