import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data

from discrete_diffusion.utils import edge_index_to_adj


class GroundTruthBackward(nn.Module):
    def __init__(self, ref_graph: Data, Qt: torch.Tensor):
        super().__init__()
        self.ref_batch = Batch.from_data_list([ref_graph])
        self.Qt = Qt

    def forward(self, x: Batch, t: torch.Tensor) -> torch.Tensor:
        """

        :param x: Batch of IDENTICAL data
        :param t: tensor of IDENTIAL times

        :return: tensor (all_possible_edges_batch, )
        """

        q_backward_all = self.backward_diffusion(x_start_batch=self.ref_batch, t_batch=t, x_t_batch=x)
        return q_backward_all[:, 1]

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
        Q_likelihood = self.Qt[0]
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
