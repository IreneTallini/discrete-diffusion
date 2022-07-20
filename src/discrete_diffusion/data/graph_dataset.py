from typing import List

import hydra
import omegaconf
from torch.utils.data import Dataset
from pathlib import Path

from nn_core.common import PROJECT_ROOT

from discrete_diffusion.data.io_utils import load_TU_dataset


class GraphDataset(Dataset):
    def __init__(self, data_dirs: List[str], dataset_names: List[str]):
        data_dirs = [Path(data_dir) for data_dir in data_dirs]
        self.samples, self.features_list = load_TU_dataset(data_dirs, dataset_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _: Dataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train,
        root=PROJECT_ROOT / "data" / "train",
        cfg=cfg.nn.data.graph_generator,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
