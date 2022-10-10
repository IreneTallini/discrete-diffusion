import logging
from typing import List, Optional, Union

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx

from discrete_diffusion.data.datamodule import MyDataModule
from discrete_diffusion.data.graph_generator import GraphGenerator

pylogger = logging.getLogger(__name__)


class SyntheticGraphDataModule(MyDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        val_percentage: float,
        graph_generator: DictConfig,
        overfit: bool,
    ):
        super().__init__(datasets, num_workers, batch_size, gpus, val_percentage)

        self.graph_generator: GraphGenerator = instantiate(config=graph_generator, _recursive_=False)

        generated_nx_graphs, self.features_list = self.graph_generator.generate_data()

        self.generated_graphs: List[Data] = [from_networkx(nx_graph) for nx_graph in generated_nx_graphs]

        ref_graph = self.generated_graphs[1]
        if overfit:
            self.generated_graphs = graph_generator.num_samples * [ref_graph]
            self.features_list = graph_generator.num_samples * [self.features_list[0]]
        self.feature_dim = ref_graph.x.shape[-1] if len(ref_graph.x.shape) > 1 else 1
        self.ref_graph_edges = ref_graph.edge_index
        self.ref_graph_feat = ref_graph.x

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
                    data_list=graphs[stage],
                )

            self.train_dataset = datasets["train"]
            self.val_datasets = [datasets["val"]]
            self.test_datasets = [datasets["test"]]

    def get_collate_fn(self, split):

        return Batch.from_data_list
