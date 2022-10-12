import logging
from typing import Any, Mapping

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers.base import DummyLogger

from discrete_diffusion.pl_modules.template_pl_module import TemplatePLModule
from discrete_diffusion.utils import (
    edge_index_to_adj,
    generate_sampled_graphs_figures,
    get_graph_sizes_from_batch,
    unflatten_batch,
)

pylogger = logging.getLogger(__name__)


class VGAEPLModule(TemplatePLModule):
    def step(self, batch) -> Mapping[str, Any]:
        z = self(batch)
        graph_sizes = get_graph_sizes_from_batch(batch)
        mask = torch.block_diag(*[torch.triu(torch.ones(i, i), diagonal=1) for i in graph_sizes]).bool()
        adj_pred_flat = dot_product_decode_batched(z, mask)

        adj = edge_index_to_adj(batch.edge_index, num_nodes=batch.num_nodes)
        adj_target = adj + torch.eye(adj.shape[0])
        adj_target_flat = adj_target[mask]

        # TODO: check next line
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        loss = norm * F.binary_cross_entropy(adj_pred_flat.view(-1), adj_target_flat.view(-1))
        adj_pred_list = unflatten_batch(adj_pred_flat, graph_sizes, len(graph_sizes))
        z_list = [z[batch.batch == i] for i in range(0, len(batch.ptr) - 1)]
        return {"loss": loss, "z_list": z_list, "adj_pred_list": adj_pred_list}

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch)
        adj_pred_list = step_out['adj_pred_list']
        z_list = step_out['z_list']
        data_list_gt = batch.to_data_list()
        if batch_idx < 1:
            fig_graph, fig_adj = generate_sampled_graphs_figures(adj_pred_list)
            fig_graph_gt, fig_adj_gt = generate_sampled_graphs_figures(data_list_gt)
            fig_z, ax_z = plt.subplots(1, 1, constrained_layout=True)
            ax_z.imshow(z_list[0].cpu().detach(), cmap="coolwarm")
            if type(self.logger) != DummyLogger:
                self.logger.log_image(key="Reconstruction/val", images=[fig_adj, fig_graph])
                self.logger.log_image(key="Ground Truth/val", images=[fig_adj_gt, fig_graph_gt])
                self.logger.log_image(key="Latent/val", images=[fig_z])
            plt.close("all")

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return step_out

    # def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
    #     step_out = self.step(batch)
    #     z = step_out["z"]
    #     batch.x = z
    #     return batch

    # def test_epoch_end(self, batch_list) -> None:
    #     data_list = []
    #     batch_size = len(batch_list[0].ptr) - 1
    #     for batch in batch_list:
    #         for i in range(batch_size):
    #             pyg_graph = get_example_from_batch(batch, i)
    #             nx_graph = pyg_to_networkx_with_features(pyg_graph)
    #             data_list.append(nx_graph)
    #     write_TU_format(data_list, path=self.hparams.latent_space_folder,
    #                     dataset_name=self.hparams.dataset_name)


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def dot_product_decode_batched(z, mask):
    similarities = torch.sigmoid(z @ z.T)
    flattened_adj_soft = similarities[mask]
    return flattened_adj_soft
