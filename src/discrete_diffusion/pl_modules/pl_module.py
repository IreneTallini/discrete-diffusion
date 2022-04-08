import logging
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import omegaconf
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch

# import torch.nn.functional as F
# import torchmetrics
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from discrete_diffusion.data.datamodule import MetaData


pylogger = logging.getLogger(__name__)


class DiffusionPLModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, model: DictConfig, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        self.model = hydra.utils.instantiate(model, feature_dim=metadata.feature_dim, _recursive_=False)

        # denoise_fn = Unet(dim=64, dim_mults=(1, 4), channels=1, out_dim=3)
        # denoise_fn = GraphDDPM(config=kwargs["cfg"])
        # self.model = Diffusion(denoise_fn=denoise_fn, image_size=28, loss_type="kl_div", timesteps=2000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.model(x)

    def step(self, x) -> Mapping[str, Any]:
        # calls forward
        loss = self(x)
        return {"loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, _ = batch
        step_out = self.step(x)

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, _ = batch
        step_out = self.step(x)

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return step_out

    def on_validation_epoch_end(self) -> None:
        # (B, C, H, W)
        batch_size_h = 2
        batch_size_w = 1
        sampled_images = self.model.sample(batch_size=batch_size_h * batch_size_w)
        sampled_images = sampled_images.permute(0, 2, 3, 1)

        fig, axs = plt.subplots(batch_size_h, batch_size_w)
        axs = axs.flatten()

        for img, ax in zip(sampled_images, axs):
            ax.imshow(img.detach().cpu())

        self.logger.experiment.log({"Sampled images": wandb.Image(fig)})

    def on_validation_epoch_end_donato(self) -> None:
        # (B, C, H, W)
        sampled_images = self.model.sample(batch_size=1)
        sampled_images = sampled_images[0]
        sampled_image = sampled_images.permute(1, 2, 0)
        sampled_image = sampled_image.squeeze(-1)
        sampled_image = (sampled_image - sampled_image.min()) / (sampled_image.max() - sampled_image.min()) * 255.0

        layout = go.Layout(
            autosize=False,
            width=500,
            height=500,
        )

        fig = go.Figure(data=go.Heatmap(z=sampled_image.detach().cpu(), zmin=0, zmax=255), layout=layout)

        wandb.log(data={"Sampled image": fig})

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, _ = batch
        step_out = self.step(x)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
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
