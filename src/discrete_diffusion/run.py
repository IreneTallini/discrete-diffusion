import logging
import math
from typing import List, Optional

import hydra
import networkx as nx
import omegaconf
import pytorch_lightning as pl
import torch_geometric.data
import wandb
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.cm import coolwarm
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import discrete_diffusion  # noqa
from discrete_diffusion.data.datamodule import MetaData
from discrete_diffusion.utils import edge_index_to_adj

pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg.train)

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run
    if fast_dev_run:
        pylogger.info(f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!")
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.trainer.gpus = 0
        cfg.nn.data.num_workers.train = 0
        cfg.nn.data.num_workers.val = 0
        cfg.nn.data.num_workers.test = 0

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.nn.data, graph_generator=cfg.nn.graph_generator, _recursive_=False
    )

    metadata: Optional[MetaData] = getattr(datamodule, "metadata", None)

    if metadata is None:
        pylogger.warning(f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>")

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.nn.module['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.module, cfg=cfg.nn, gpus=cfg.train.trainer.gpus, _recursive_=False, metadata=metadata
    )

    # Instantiate the callbacks
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)
    pylogger.info("Instantiating the <Trainer>")

    ref_batch = next(iter(datamodule.train_dataloader()))
    ref_list = torch_geometric.data.Batch.to_data_list(ref_batch)
    side = math.ceil(math.sqrt(cfg.nn.data.batch_size.train))
    fig, axes = plt.subplots(side, side, constrained_layout=True)
    fig_feat, axes_feat = plt.subplots(side, side, constrained_layout=True)

    if side > 1:
        axs = axes.flatten()
        axs_feat = axes_feat.flatten()
    else:
        axs = [axes]
        axs_feat = [axes_feat]

    for i in range(cfg.nn.data.batch_size.train):
        graph = ref_list[i]
        nx_graph = torch_geometric.utils.to_networkx(graph)
        nx.draw(nx_graph, with_labels=True, ax=axs[i], node_size=1)
        vmin = int(graph.x.min())
        vmax = int(graph.x.max())
        axs_feat[i].imshow(graph.x.cpu().detach(), vmin=vmin, vmax=vmax, cmap=coolwarm)
        # axs_feat[i].colorbar(cm.ScalarMappable(cmap=coolwarm))
    cbar = fig_feat.colorbar(cm.ScalarMappable(cmap=coolwarm), ticks=[0, 1])
    cbar.ax.set_yticklabels([vmin, vmax])

    logger.experiment.log({"Dataset Example": wandb.Image(fig)})
    logger.experiment.log({"Dataset Features Example": wandb.Image(fig_feat)})
    plt.close()

    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=template_core.trainer_ckpt_path)

    if fast_dev_run:
        pylogger.info("Skipping testing in 'fast_dev_run' mode!")
    else:
        if "test" in cfg.nn.data.datasets and trainer.checkpoint_callback.best_model_path is not None:
            pylogger.info("Starting testing!")
            trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
