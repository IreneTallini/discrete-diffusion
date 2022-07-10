import logging
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers.base import DummyLogger
from torch.optim import Optimizer
from torch_geometric.data import Batch

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from discrete_diffusion.data.datamodule import MetaData
from discrete_diffusion.pl_modules.pl_module import TemplatePLModule
from discrete_diffusion.utils import clear_figures, generate_sampled_graphs_figures

pylogger = logging.getLogger(__name__)

class DiffusionPLModule(TemplatePLModule):
    def __init__(self, model: DictConfig, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(model, metadata=metadata, *args, **kwargs)

        self.num_nodes_samples = self.set_nodes_number_sampling()

    def set_nodes_number_sampling(self) -> int:
        num_nodes_list = torch.tensor([len(feature) for feature in self.metadata.features_list])
        if self.hparams.num_nodes_samples == -1:
            idx = torch.randint(len(num_nodes_list), (1,), dtype=torch.int64)
            num_nodes_samples = int(num_nodes_list.index_select(dim=0, index=idx))
        elif self.hparams.num_nodes_samples > 0:
            num_nodes_samples = self.hparams.num_nodes_samples
        else:
            raise AttributeError("num_nodes_samples should be -1 (random choice) or n > 0")
        return num_nodes_samples

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

        clear_figures([fig, fig_adj, diffusion_images])

    def on_test_epoch_end(self) -> None:
        sampled_graphs, _ = self.sample_from_model([self.num_nodes_samples] * self.hparams.batch_size.test)
        fig, fig_adj = generate_sampled_graphs_figures(sampled_graphs)
        if type(self.logger) != DummyLogger:
            self.logger.log_image(key="Sampled graphs/test", images=[fig])
            self.logger.log_image(key="Sampled adj/test", images=[fig_adj])

        clear_figures([fig, fig_adj])

    def sample_from_model(self, num_nodes_samples: List[int]) -> (Batch, plt.Figure):
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
