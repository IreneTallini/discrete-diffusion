import logging
from typing import Any, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric.utils
from pytorch_lightning.loggers.base import DummyLogger

from discrete_diffusion.pl_modules.template_pl_module import TemplatePLModule
from discrete_diffusion.utils import (
    adj_to_edge_index,
    edge_index_to_adj,
    get_data_from_edge_index,
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
        return {"loss": loss, "z": z, "adj_pred_list": adj_pred_list}

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch)
        A_pred = step_out['A_pred']
        if batch_idx < 5:
            fig, axs = plt.subplots(2, 2, constrained_layout=True)
            gt_adjs = edge_index_to_adj(batch.edge_index, len(batch.batch))
            im = axs[0, 0].imshow(A_pred.T.cpu(), cmap='coolwarm')
            axs[0, 0].set_title("reconstruction")
            plt.colorbar(im, ax=axs[0, 0], orientation='vertical')
            axs[0, 1].imshow(gt_adjs.T.cpu())
            axs[0, 1].set_title("ground truth")

            disc_adj = (A_pred > 0.5).long() - torch.eye(A_pred.shape[0])
            edge_index = adj_to_edge_index(disc_adj)
            data = get_data_from_edge_index(edge_index, batch.x)
            nx.draw(torch_geometric.utils.to_networkx(data), with_labels=True, ax=axs[1, 0], node_size=0.1)
            nx.draw(torch_geometric.utils.to_networkx(batch), with_labels=True, ax=axs[1, 1], node_size=0.1)
            if type(self.logger) != DummyLogger:
                self.logger.log_image(key="Reconstruction/val", images=[fig])
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
