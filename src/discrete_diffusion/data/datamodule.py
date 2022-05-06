import json
import logging
from functools import cached_property, partial
from pathlib import Path
from typing import List, Optional, Sequence, Union

import hydra
import networkx as nx
import omegaconf
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx
from torchvision import transforms

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

from discrete_diffusion.data.graph_generator import GraphGenerator
from discrete_diffusion.data.io_utils import load_data_irene, random_split_sequence

# from src.discrete_diffusion.utils import edge_index_to_adj


pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, feature_dim, train_data):
        """The data information the Lightning Module will be provided with.
        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.
        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.
        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.
        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.
        Args:
            feature_dim
        """
        self.feature_dim = feature_dim
        self.train_data = train_data

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.
        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        data = {
            "feature_dim": self.feature_dim,
            # "train_data": self.train_data,
        }

        (dst_path / "data.json").write_text(json.dumps(data, indent=4, default=lambda x: x.__dict__))

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.
        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint
        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        data = json.loads((src_path / "data.json").read_text(encoding="utf-8"))

        return MetaData(
            feature_dim=data["feature_dim"],
            # train_data=data["train_data"],
        )


def collate_fn(samples: List, split: Split, metadata: MetaData):
    """Custom collate function for dataloaders with access to split and metadata.

    Args:
        samples: A list of samples coming from the Dataset to be merged into a batch
        split: The data split (e.g. train/val/test)
        metadata: The MetaData instance coming from the DataModule or the restored checkpoint

    Returns:
        A batch generated from the given samples
    """
    return default_collate(samples)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        val_percentage: float,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"
        self.feature_dim = 0

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        # example
        self.val_percentage: float = val_percentage

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        return MetaData(feature_dim=self.feature_dim, train_data=self.train_dataset)

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):
            # example
            mnist_train = hydra.utils.instantiate(
                self.datasets.train,
                split="train",
                transform=transform,
                path=PROJECT_ROOT / "data",
            )
            train_length = int(len(mnist_train) * (1 - self.val_percentage))
            val_length = len(mnist_train) - train_length
            self.train_dataset, val_dataset = random_split(
                mnist_train, [train_length, val_length], generator=torch.Generator().manual_seed(1)
            )

            self.val_datasets = [val_dataset]

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(
                    dataset_cfg,
                    split="test",
                    path=PROJECT_ROOT / "data",
                    transform=transform,
                )
                for dataset_cfg in self.datasets.test
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn("train"),
        )

    def get_collate_fn(self, split):
        return partial(collate_fn, split=split, metadata=self.metadata)

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
                collate_fn=self.get_collate_fn("val"),
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
                collate_fn=self.get_collate_fn("test"),
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


class SyntheticGraphDataModule(MyDataModule):
    def __init__(
        self,
        graph_generator: DictConfig,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        val_percentage: float,
    ):
        super().__init__(datasets, num_workers, batch_size, gpus, val_percentage)

        self.graph_generator: GraphGenerator = instantiate(config=graph_generator, _recursive_=False)

        generated_nx_graphs: List[nx.Graph] = self.graph_generator.generate_data()

        self.generated_graphs: List[Data] = [from_networkx(nx_graph) for nx_graph in generated_nx_graphs]

        ref_graph = self.generated_graphs[0]
        self.feature_dim = ref_graph.x.shape[-1] if len(ref_graph.x.shape) > 1 else 1

    def setup(self, stage: Optional[str] = None):

        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):

            graphs = {}
            graphs["train"], graphs["val"], graphs["test"] = self.split_train_val_test(self.generated_graphs)

            stages = {"train", "val", "test"}
            datasets = {}

            for stage in stages:
                config = self.datasets[stage]
                datasets[stage] = hydra.utils.instantiate(
                    config=config,
                    samples=graphs[stage],
                )

            self.train_dataset = datasets["train"]
            self.val_datasets = [datasets["val"]]
            self.test_datasets = [datasets["test"]]

    def split_train_val_test(self, graphs):
        split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}

        train_val, test = random_split_sequence(graphs, split_ratio["train"] + split_ratio["val"])

        train, val = random_split_sequence(
            train_val, split_ratio["train"] / (split_ratio["train"] + split_ratio["val"])
        )

        return train, val, test

    def get_collate_fn(self, split):

        return Batch.from_data_list


class GraphDataModule(MyDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        feature_params,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        val_percentage: float,
        **kwargs,
    ):
        super().__init__(datasets, num_workers, batch_size, gpus, val_percentage)
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        self.data_list, self.class_to_label_dict = load_data_irene(
            self.data_dir,
            self.dataset_name,
            feature_params=feature_params,
        )

        ref_data = self.data_list[0]
        self.feature_dim = ref_data.x.shape[-1] if len(ref_data.x.shape) > 1 else 1

    def setup(self, stage: Optional[str] = None):

        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):

            graphs = {}
            graphs["train"], graphs["val"], graphs["test"] = self.split_train_val_test(self.data_list)

            stages = {"train", "val", "test"}
            datasets = {}

            for stage in stages:
                config = self.datasets[stage]
                datasets[stage] = hydra.utils.instantiate(
                    config=config,
                    samples=graphs[stage],
                )

            self.train_dataset = datasets["train"]
            self.val_datasets = [datasets["val"]]
            self.test_datasets = [datasets["test"]]

    def split_train_val_test(self, graphs):
        split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}

        train_val, test = random_split_sequence(graphs, split_ratio["train"] + split_ratio["val"])

        train, val = random_split_sequence(
            train_val, split_ratio["train"] / (split_ratio["train"] + split_ratio["val"])
        )

        return train, val, test

    def get_collate_fn(self, split):

        return Batch.from_data_list


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)


if __name__ == "__main__":
    main()
