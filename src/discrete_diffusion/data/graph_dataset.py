import hydra
import omegaconf
from torch.utils.data import Dataset

from nn_core.common import PROJECT_ROOT


class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.samples = data_list

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
