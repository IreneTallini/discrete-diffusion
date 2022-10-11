import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.base import DummyLogger
from torch.optim import Optimizer

from nn_core.model_logging import NNLogger
from nn_core.serialization import load_model

from discrete_diffusion.data.datamodule import MetaData
from discrete_diffusion.pl_modules.vgae_pl_module import dot_product_decode
from discrete_diffusion.utils import (
    adj_to_edge_index,
    clear_figures,
    generate_sampled_graphs_figures,
    generate_sampled_latents_figures,
    get_data_from_edge_index,
)

pylogger = logging.getLogger(__name__)


class LatentDiffusionPLModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()
        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.metadata = metadata
        self.node_embedder = hydra.utils.instantiate(self.hparams.node_embedder, feature_dim=metadata.feature_dim,
                                                     _recursive_=False)
        # self.vgae.load
        node_embedder_state_dict = torch.load(self.hparams.node_embedder_checkpoint_path)
        self.node_embedder.load_state_dict({k.replace('model.', ''):v for k, v in node_embedder_state_dict['state_dict'].items()})
        self.diffusion = hydra.utils.instantiate(self.hparams.diffusion, feature_len=len(metadata.features_list[0]),
                                                 _recursive_=False)
        # self.num_nodes_samples = self.set_nodes_number_sampling()

    def step(self, x) -> Mapping[str, Any]:
        with torch.no_grad():
            z = self.node_embedder(x)
        # z = z.unsqueeze(-1)
        z = z.unsqueeze(0)
        z = z.permute(0, 2, 1)
        z = z.unsqueeze(-1)
        loss = self.diffusion(z)
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

    # def set_nodes_number_sampling(self) -> int:
    #     num_nodes_list = torch.tensor([len(feature) for feature in self.metadata.features_list])
    #     if self.hparams.num_nodes_samples == -1:
    #         idx = torch.randint(len(num_nodes_list), (1,), dtype=torch.int64)
    #         num_nodes_samples = int(num_nodes_list.index_select(dim=0, index=idx))
    #     elif self.hparams.num_nodes_samples > 0:
    #         num_nodes_samples = self.hparams.num_nodes_samples
    #     else:
    #         raise AttributeError("num_nodes_samples should be -1 (random choice) or n > 0")
    #     return num_nodes_samples

    def on_validation_epoch_end(self) -> None:
        sampled_latents = self.diffusion.sample(device=self.device)
        sampled_latents = sampled_latents.squeeze(-1)
        sampled_latents = sampled_latents.permute(0, 2, 1)
        sampled_latents = sampled_latents.squeeze(0)

        sampled_adj = (torch.nn.functional.sigmoid(sampled_latents @ sampled_latents.T) > 0.5).long()
        sampled_edge_idx = adj_to_edge_index(sampled_adj)
        sampled_data = get_data_from_edge_index(sampled_edge_idx, node_features=torch.empty((sampled_adj.shape[0], 1)))
        fig_graph, fig_adj = generate_sampled_graphs_figures([sampled_data])

        fig_latent = generate_sampled_latents_figures(sampled_latents.unsqueeze(0))
        if type(self.logger) != DummyLogger:
            self.logger.log_image(key="Sampled latent/val", images=[fig_latent])
            self.logger.log_image(key="Sampled adj/val", images=[fig_adj])
            self.logger.log_image(key="Sampled graph/val", images=[fig_graph])
        clear_figures([fig_latent, fig_adj, fig_graph])

    # def on_test_epoch_end(self) -> None:
    #     sampled_graphs, _ = self.sample_from_model([self.num_nodes_samples] * self.hparams.batch_size.test)
    #     fig, fig_adj = generate_sampled_graphs_figures(sampled_graphs)
    #     if type(self.logger) != DummyLogger:
    #         self.logger.log_image(key="Sampled graphs/test", images=[fig])
    #         self.logger.log_image(key="Sampled adj/test", images=[fig_adj])
    #
    #     clear_figures([fig, fig_adj])

    # def sample_from_model(self, num_nodes_samples: List[int]) -> (Batch, plt.Figure):
    #     sampled_graphs, diffusion_images = self.model.sample(
    #         self.metadata.features_list, num_nodes_samples=num_nodes_samples
    #     )
    #     return sampled_graphs, diffusion_images

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
