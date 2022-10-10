import json
import logging
from functools import cached_property, partial
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from nn_core.nn_types import Split

from discrete_diffusion.io_utils import random_split_sequence

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, feature_dim, features_list, ref_graph_edges, ref_graph_feat):
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
        self.features_list = features_list
        self.ref_graph_edges = ref_graph_edges
        self.ref_graph_feat = ref_graph_feat

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.
        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        data = {
            "feature_dim": self.feature_dim,
            "features_list": self.features_list,
            "ref_graph_edges": self.ref_graph_edges,
            "ref_graph_feat": self.ref_graph_feat,
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
            features_list=data["features_list"],
            ref_graph_feat=data["ref_graph_feat"],
            ref_graph_edges=data["ref_graph_edges"],
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

        return MetaData(
            feature_dim=self.feature_dim,
            features_list=self.features_list,
            ref_graph_edges=self.ref_graph_edges,
            ref_graph_feat=self.ref_graph_feat,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn("train"),
            # multiprocessing_context="fork",
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
                # multiprocessing_context="fork",
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
                # multiprocessing_context="fork",
            )
            for dataset in self.test_datasets
        ]

    @staticmethod
    def split_train_val_test(graphs):
        split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}

        train_val, test = random_split_sequence(graphs, split_ratio["train"] + split_ratio["val"])

        train, val = random_split_sequence(
            train_val, split_ratio["train"] / (split_ratio["train"] + split_ratio["val"])
        )

        return train, val, test

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"
