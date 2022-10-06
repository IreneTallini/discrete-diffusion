import logging
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import pytorch_lightning as pl
import torch
import torch_geometric.utils
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer

from nn_core.model_logging import NNLogger

from discrete_diffusion.data.datamodule import MetaData
from discrete_diffusion.utils import adj_to_edge_index, edge_index_to_adj, get_data_from_edge_index

pylogger = logging.getLogger(__name__)


class VGAEPLModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, model: DictConfig, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.metadata = metadata
        self.model = self.instantiate_model(model, metadata)

    def instantiate_model(self, model, metadata):
        inst_model = hydra.utils.instantiate(model, feature_dim=metadata.feature_dim, _recursive_=False)
        return inst_model

    def forward(self, x: Any) -> torch.Tensor:
        """Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        return self.model(x)

    def step(self, x) -> Mapping[str, Any]:
        loss = self(x)
        return {"loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch)
        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch)
        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return step_out

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch)
        self.log_dict(
            {
                "loss/test": step_out["loss"].cpu().detach(),
            },
        )

        return step_out

    def configure_optimizers(
            self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return opt
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]

    # def step(self, batch) -> Mapping[str, Any]:
    #     loss, z, x_rec = self(batch)
    #     # edge_index = batch.edge_index
    #     # adj_matrix = edge_index_to_adj(edge_index, num_nodes=10).to(torch.float32)  # .to(self.device)
    #
    #     # graph_sizes = get_graph_sizes_from_batch(batch)
    #     # mask = torch.block_diag(*[torch.triu(torch.ones(i, i), diagonal=1) for i in graph_sizes]).bool()
    #
    #     # loss = torch.norm(gt_adjs[mask] - similarities[mask]) ** 2
    #     # loss = torch.sum(torch.abs(gt_adjs - similarities))
    #     # loss = torch.binary_cross_entropy_with_logits(similarities[mask], gt_adjs[mask].float()).mean()
    #     return {"loss": loss, "z": z, "x_rec": x_rec}

    # def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
    #     logger: NNLogger
    #     step_out = self.step(batch)
    #     z = step_out['z']
    #     x_rec = step_out['x_rec']
    #     # z = step_out["z"]
    #     # similarities = z @ z.T
    #     if batch_idx < 5:
    #         fig, axs = plt.subplots(2, 2, constrained_layout=True)
    #         gt_adjs = edge_index_to_adj(batch.edge_index, len(batch.batch))
    #         im = axs[0, 0].imshow(x_rec.T.cpu(), cmap='coolwarm')
    #         axs[0, 0].set_title("reconstruction")
    #         plt.colorbar(im, ax=axs[0, 0], orientation='vertical')
    #         axs[0, 1].imshow(gt_adjs.T.cpu())
    #         axs[0, 1].set_title("ground truth")
    #
    #         disc_adj = (x_rec > 0.5).long()
    #         edge_index = adj_to_edge_index(disc_adj)
    #         data = get_data_from_edge_index(edge_index, batch.x)
    #         nx.draw(torch_geometric.utils.to_networkx(data), with_labels=True, ax=axs[1, 0], node_size=0.1)
    #         nx.draw(torch_geometric.utils.to_networkx(batch), with_labels=True, ax=axs[1, 1], node_size=0.1)
    #         wandb.log({"Reconstruction Example": wandb.Image(fig)})
    #
    #     self.log_dict(
    #         {"loss/val": step_out["loss"].cpu().detach()},
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=True,
    #     )
    #     return step_out

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
