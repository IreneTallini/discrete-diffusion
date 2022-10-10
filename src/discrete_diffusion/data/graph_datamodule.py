import logging
from pathlib import Path
from typing import List, Optional, Union

import hydra
from omegaconf import DictConfig
from torch_geometric.data import Batch

from discrete_diffusion.data.datamodule import MyDataModule
from discrete_diffusion.io_utils import load_TU_dataset, random_split_sequence

pylogger = logging.getLogger(__name__)


class GraphDataModule(MyDataModule):
    def __init__(
        self,
        dataset_name: str,
        datasets: DictConfig,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        max_graphs_per_dataset: int,
        max_num_nodes: int,
        min_num_nodes: int,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        val_percentage: float,
        overfit: bool,
        **kwargs,
    ):
        super().__init__(datasets, num_workers, batch_size, gpus, val_percentage)
        self.datasets = datasets
        self.overfit = overfit
        self.dataset_name = dataset_name
        self.max_graphs_per_dataset = max_graphs_per_dataset
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.max_num_nodes = max_num_nodes
        self.min_num_nodes = min_num_nodes

    def setup(self, stage: Optional[str] = None):
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):

            data_list, features_list = load_TU_dataset(
                paths=[Path(self.train_dir) / self.dataset_name],
                dataset_names=[self.dataset_name],
                max_graphs_per_dataset=[self.max_graphs_per_dataset],
                max_num_nodes=self.max_num_nodes,
                min_num_nodes=self.min_num_nodes
            )

            ref_graph = data_list[0]
            self.features_list = features_list
            if self.overfit > -1:
                data_list = data_list[: self.overfit]
                features_list = self.features_list[: self.overfit]
                multiplicity = len(self.features_list) // self.overfit + 1
                data_list = multiplicity * data_list
                self.features_list = multiplicity * features_list
            self.feature_dim = len(ref_graph.x[0]) if len(ref_graph.x[0]) > 1 else 1
            self.ref_graph_edges = ref_graph.edge_index
            self.ref_graph_feat = ref_graph.x

            self.train_dataset = hydra.utils.instantiate(config=self.datasets["train"], data_list=data_list)

            if self.overfit > -1:
                self.val_datasets = [hydra.utils.instantiate(config=self.datasets["train"],
                                                             data_list=data_list[:int(0.1 * len(data_list))])]
                self.test_datasets = [hydra.utils.instantiate(config=self.datasets["train"],
                                                              data_list=data_list[:int(0.1 * len(data_list))])]
            else:
                val_data_list, _ = load_TU_dataset(
                    paths=[Path(self.val_dir) / self.dataset_name],
                    dataset_names=[self.dataset_name],
                    max_graphs_per_dataset=[0.1 * len(data_list)],
                    max_num_nodes=self.max_num_nodes,
                    min_num_nodes=self.min_num_nodes
                )

                self.val_datasets = [hydra.utils.instantiate(config=self.datasets["val"], data_list=val_data_list)]

                test_data_list, _ = load_TU_dataset(
                    paths=[Path(self.test_dir) / self.dataset_name],
                    dataset_names=[self.dataset_name],
                    max_graphs_per_dataset=[0.1 * len(data_list)],
                    max_num_nodes=self.max_num_nodes,
                    min_num_nodes=self.min_num_nodes
                )

                self.test_datasets = [hydra.utils.instantiate(config=self.datasets["test"], data_list=test_data_list)]

    @staticmethod
    def split_train_val_test(graphs):
        split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}

        train_val, test = random_split_sequence(graphs, split_ratio["train"] + split_ratio["val"])

        train, val = random_split_sequence(
            train_val, split_ratio["train"] / (split_ratio["train"] + split_ratio["val"])
        )

        return train, val, test

    def get_collate_fn(self, split):

        return Batch.from_data_list
