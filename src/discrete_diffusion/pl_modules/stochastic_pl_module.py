import csv
import logging
import pathlib
from typing import Any, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.utils
from pytorch_lightning.loggers.base import DummyLogger

from nn_core.model_logging import NNLogger

from discrete_diffusion.io_utils import write_TU_format
from discrete_diffusion.pl_modules.pl_module import TemplatePLModule
from discrete_diffusion.utils import (
    adj_to_edge_index,
    edge_index_to_adj,
    generate_sampled_graphs_figures,
    get_data_from_edge_index,
    get_example_from_batch,
    get_graph_sizes_from_batch,
    pyg_to_networkx_with_features,
    unflatten_adj,
)

pylogger = logging.getLogger(__name__)


class StochasticPLModule(TemplatePLModule):

    def step(self, batch) -> Mapping[str, Any]:
        loss, z = self(batch)
        # edge_index = batch.edge_index
        # adj_matrix = edge_index_to_adj(edge_index, num_nodes=10).to(torch.float32)  # .to(self.device)

        # graph_sizes = get_graph_sizes_from_batch(batch)
        # mask = torch.block_diag(*[torch.triu(torch.ones(i, i), diagonal=1) for i in graph_sizes]).bool()

        # loss = torch.norm(gt_adjs[mask] - similarities[mask]) ** 2
        # loss = torch.sum(torch.abs(gt_adjs - similarities))
        # loss = torch.binary_cross_entropy_with_logits(similarities[mask], gt_adjs[mask].float()).mean()
        return {"loss": loss, "z": z}

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:

        step_out = self.step(batch)
        z = step_out["z"]

        gt_data = batch
        out_data = gt_data
        if batch_idx < 16:
            x = self.model.decode_sample(z, n_samples=1)
            out_A = unflatten_adj(x, num_nodes=self.model.max_num_nodes)
            # plt.imshow(out_A.cpu())

            # plt.savefig('gen.png')
            # plt.close()

            edge_out_A = adj_to_edge_index(out_A)
            out_data = get_data_from_edge_index(edge_out_A, batch.x)

            # nx.draw(torch_geometric.utils.to_networkx(out_data), with_labels=True, node_size=0.1)
            # plt.savefig('gen_graph.png')
            # wandb.log({"Reconstruction Example": wandb.Image(fig)})
            # plt.close()
            # input_A = edge_index_to_adj(batch.edge_index, num_nodes=self.model.max_num_nodes)
            # plt.imshow(input_A.cpu())
            # plt.savefig('input.png')
            # plt.close()
            # nx.draw(torch_geometric.utils.to_networkx(batch), with_labels=True,  node_size=0.1)
            # plt.savefig('input_graph.png')
            # plt.close()

    # z = step_out['z']
        # x_rec = step_out['x_rec']
        # z = step_out["z"]
        # similarities = z @ z.T
        # if batch_idx < 5:
        #     fig, axs = plt.subplots(2, 2, constrained_layout=True)
        #     gt_adjs = edge_index_to_adj(batch.edge_index, len(batch.batch))
        #     im = axs[0, 0].imshow(x_rec.T.cpu(), cmap='coolwarm')
        #     axs[0, 0].set_title("reconstruction")
        #     plt.colorbar(im, ax=axs[0, 0], orientation='vertical')
        #     axs[0, 1].imshow(gt_adjs.T.cpu())
        #     axs[0, 1].set_title("ground truth")
        #
        #     disc_adj = (x_rec > 0.5).long()
        #     edge_index = adj_to_edge_index(disc_adj)
        #     data = get_data_from_edge_index(edge_index, batch.x)
        #     nx.draw(torch_geometric.utils.to_networkx(data), with_labels=True, ax=axs[1, 0], node_size=0.1)
        #     nx.draw(torch_geometric.utils.to_networkx(batch), with_labels=True, ax=axs[1, 1], node_size=0.1)
        #     wandb.log({"Reconstruction Example": wandb.Image(fig)})

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return out_data

    def validation_epoch_end(self, outputs) -> None:
        fig, fig_adj = generate_sampled_graphs_figures(outputs[:16])
        if type(self.logger) != DummyLogger:
            self.logger.log_image(key="Sampled Graphs", images=[fig])
            self.logger.log_image(key="Sampled Adj", images=[fig_adj])

    # def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
    #     step_out = self.step(batch)
    #     z = step_out["z"]
    #     batch.x = z
    #     return batch
    #
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
