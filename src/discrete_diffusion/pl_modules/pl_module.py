import logging
import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import omegaconf
import pytorch_lightning as pl
import torch
import torch_geometric.utils
from omegaconf import DictConfig
from pytorch_lightning.loggers.base import DummyLogger
from torch.optim import Optimizer
from torch_geometric.data import Data

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from discrete_diffusion.data.datamodule import MetaData
from discrete_diffusion.utils import edge_index_to_adj, generate_sampled_graphs_figures

pylogger = logging.getLogger(__name__)


class TemplatePLModule(pl.LightningModule):
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
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]


class DiffusionPLModule(TemplatePLModule):
    def __init__(self, model: DictConfig, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(model, metadata=metadata, *args, **kwargs)

        num_nodes_list = torch.tensor([len(feature) for feature in self.metadata.features_list])
        if self.hparams.type_num_nodes_samples == -1:
            idx = torch.randint(len(num_nodes_list), (1,), dtype=torch.int64)
            self.num_nodes_samples = int(num_nodes_list.index_select(dim=0, index=idx))
        elif self.hparams.type_num_nodes_samples > 0:
            self.num_nodes_samples = self.hparams.type_num_nodes_samples
        else:
            raise AttributeError("type_num_nodes_samples should be -1 (random choice) or n > 0")

    def on_validation_epoch_end(self) -> None:
        sampled_graphs, diffusion_images = self.sample_from_model(
            [self.num_nodes_samples] * self.hparams.batch_size.val
        )

        fig, fig_adj = generate_sampled_graphs_figures(sampled_graphs)

        if type(self.logger) != DummyLogger:
            self.logger.log_image(key="Sampled graphs/val", images=[fig])
            self.logger.log_image(key="Sampled adj/val", images=[fig_adj])
            self.logger.log_image(
                key="Adjacency matrices sampling process for first graph in batch/val", images=[diffusion_images]
            )

        plt.close(fig)
        plt.close(diffusion_images)

    def on_test_epoch_end(self) -> None:
        sampled_graphs, _ = self.sample_from_model([self.num_nodes_samples] * self.hparams.batch_size.test)
        fig, fig_adj = generate_sampled_graphs_figures(sampled_graphs)
        if type(self.logger) != DummyLogger:
            self.logger.log_image(key="Sampled graphs/test", images=[fig])
            self.logger.log_image(key="Sampled adj/test", images=[fig_adj])

    def sample_from_model(self, num_nodes_samples):
        sampled_graphs, diffusion_images = self.model.sample(
            self.metadata.features_list, num_nodes_samples=num_nodes_samples
        )
        return sampled_graphs, diffusion_images


class GroundTruthDiffusionPLModule(DiffusionPLModule):
    logger: NNLogger

    def instantiate_model(self, model, metadata):
        self.register_buffer("ref_graph_edges", self.metadata.ref_graph_edges)
        self.register_buffer("ref_graph_feat", self.metadata.ref_graph_feat)

        inst_model = hydra.utils.instantiate(
            model,
            feature_dim=metadata.feature_dim,
            ref_graph_edges=self.ref_graph_edges,
            ref_graph_feat=self.ref_graph_feat,
            _recursive_=False,
        )
        return inst_model


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
