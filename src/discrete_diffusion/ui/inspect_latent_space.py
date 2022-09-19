

data_list = []
batch_size = len(batch_list[0].ptr) - 1
for batch in batch_list:
    for i in range(batch_size):
        pyg_graph = get_example_from_batch(batch, i)
        nx_graph = pyg_to_networkx_with_features(pyg_graph)
        data_list.append(nx_graph)
write_TU_format(data_list, path=self.hparams.latent_space_folder,
                dataset_name=self.hparams.dataset_name)
