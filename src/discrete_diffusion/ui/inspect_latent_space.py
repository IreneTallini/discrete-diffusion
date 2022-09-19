import hydra
import omegaconf

from nn_core.common import PROJECT_ROOT

from discrete_diffusion.io_utils import write_TU_format
from discrete_diffusion.utils import get_example_from_batch, pyg_to_networkx_with_features


def run(cfg):
    ...
#     test_datasets = [hydra.utils.instantiate(config=,
#                                                  data_list=data_list[:int(0.1 * len(data_list))])]

#    data_list = []
#    batch_size = len(batch_list[0].ptr) - 1
#    for batch in batch_list:
#        for i in range(batch_size):
#            pyg_graph = get_example_from_batch(batch, i)
#            nx_graph = pyg_to_networkx_with_features(pyg_graph)
#            data_list.append(nx_graph)
#    write_TU_format(data_list, path=cfg.nn.module.latent_space_folder,
#                    dataset_name=cfg.nn.module.hparams.dataset_name)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
