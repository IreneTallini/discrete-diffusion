import logging
from typing import Any, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric.utils
from pytorch_lightning.loggers.base import DummyLogger

from discrete_diffusion.pl_modules.template_pl_module import TemplatePLModule
from discrete_diffusion.utils import adj_to_edge_index, edge_index_to_adj, get_data_from_edge_index

pylogger = logging.getLogger(__name__)


class VGAEPLModule(TemplatePLModule):
    def step(self, batch) -> Mapping[str, Any]:
        z = self(batch)
        A_pred = dot_product_decode(z)
        adj = edge_index_to_adj(batch.edge_index-1, num_nodes=10)
        adj_target = adj + torch.eye(adj.shape[0])
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_target.view(-1))
        return {"loss": loss, "z": z, "A_pred": A_pred}

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
