from copy import copy

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import Batch

from discrete_diffusion.modules.diffusion import Diffusion
from discrete_diffusion.utils import adj_to_edge_index, edge_index_to_adj


class DiscreteDiffusion(Diffusion):
    def __init__(
        self,
        denoise_fn: DictConfig,
        feature_dim: int,
        diffusion_speed: float,
        timesteps: int,
        threshold_sample: float,
    ):
        super().__init__(denoise_fn, feature_dim, diffusion_speed, timesteps, threshold_sample)

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
        cross_ent = F.binary_cross_entropy(q_approx, q_target, reduction="mean")
        ent = F.binary_cross_entropy(q_target, q_target, reduction="mean")
        loss = cross_ent - ent
        return loss


class GroundTruthDiscreteDiffusion(DiscreteDiffusion):
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
