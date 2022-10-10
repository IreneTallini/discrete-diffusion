import hydra.utils
import torch.nn as nn
import torch_geometric
from omegaconf import DictConfig
from torch import repeat_interleave
from torch_geometric.nn import JumpingKnowledge

from discrete_diffusion.modules.gnn.mlp import MLP
from discrete_diffusion.utils import get_graph_sizes_from_batch


class NodeEmbedder(nn.Module):
    def __init__(
        self,
        gnn: DictConfig,
        feature_dim,
        embedding_dim,
        hidden_dim_shared,
        use_batch_norm_pre_post_mlp,
        num_layers_pre_post_mlp,
        num_convs,
        dropout_rate,
        do_preprocess,
        jump_mode="cat",
        do_time_conditioning=False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim_shared = hidden_dim_shared
        self.use_batch_norm_pre_post_mlp = use_batch_norm_pre_post_mlp
        self.embedding_dim = embedding_dim
        self.num_convs = num_convs
        self.jump_mode = jump_mode
        self.do_preprocess = do_preprocess
        self.do_time_conditioning = do_time_conditioning

        self.preprocess_mlp = (
            MLP(
                num_layers=num_layers_pre_post_mlp,
                input_dim=self.feature_dim,
                output_dim=self.hidden_dim_shared,
                hidden_dim=self.hidden_dim_shared,
                use_batch_norm=self.use_batch_norm_pre_post_mlp,
            )
            if do_preprocess
            else None
        )

        self.convs = nn.ModuleList()
        for conv in range(self.num_convs):
            input_dim = self.feature_dim if (conv == 0 and not self.do_preprocess) else self.hidden_dim_shared
            output_dim = self.hidden_dim_shared
            conv = hydra.utils.instantiate(gnn, input_dim=input_dim, output_dim=output_dim)
            self.convs.append(conv)

        if self.do_time_conditioning:
            self.time_mlps = nn.ModuleList()
            for time_mlp in range(self.num_convs):
                output_dim = self.embedding_dim if (time_mlp == self.num_convs - 1) else self.hidden_dim_shared
                time_mlp = MLP(
                    num_layers=num_layers_pre_post_mlp,
                    input_dim=1,
                    output_dim=output_dim,
                    hidden_dim=self.hidden_dim_pre_post_mlp,
                    use_batch_norm=self.use_batch_norm_pre_post_mlp,
                )
                self.time_mlps.append(time_mlp)

        num_layers = (self.num_convs + 1) if self.do_preprocess else self.num_convs
        if self.jump_mode == "cat":
            pooled_dim = (num_layers * self.hidden_dim_shared) + self.feature_dim
        else:
            pooled_dim = self.hidden_dim_shared

        self.dropout = nn.Dropout(p=dropout_rate)

        self.postprocess_mlp = MLP(
            num_layers=num_layers_pre_post_mlp,
            input_dim=pooled_dim,
            output_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim_shared,
            use_batch_norm=self.use_batch_norm_pre_post_mlp,
        )

        self.jumping_knowledge = JumpingKnowledge(mode=self.jump_mode) if self.jump_mode != "none" else None

    def forward(self, batch, t=None):
        """
        Embeds a batch of graphs given as a single large graph
        :param batch: Batch containing graphs to embed
        :param t: timestep
        :return: embedded graphs, each graph embedded as a point in R^{E}
        """

        # X ~ (num_nodes_in_batch, feature_dim)
        # edge_index ~ (2, num_edges_in_batch)
        X, edge_index = batch.x, batch.edge_index

        if len(X.shape) == 1:
            X = X.unsqueeze(-1)

        h = self.preprocess_mlp(X) if self.do_preprocess else X
        jump_xs = [X, h] if self.do_preprocess else [X]

        graph_sizes = get_graph_sizes_from_batch(batch)
        for conv_step, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if self.do_time_conditioning:
                time_mlp = self.time_mlps[conv_step]
                time_emb = time_mlp(t[:, None].float())
                time_emb = repeat_interleave(time_emb, graph_sizes, dim=0)
                h = h + time_emb
            jump_xs.append(h)

        if self.jump_mode != "none":
            h = self.jumping_knowledge(jump_xs)

        h = self.dropout(h)
        # out ~ (num_nodes_in_batch, output_dim)
        node_out_features = self.postprocess_mlp(h)

        return node_out_features


class GATWrapper(torch_geometric.nn.GATConv):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super(GATWrapper, self).__init__(in_channels=input_dim, out_channels=output_dim, **kwargs)


class GINWrapper(torch_geometric.nn.GINConv):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):

        mlp = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                **kwargs,
            )
        super(GINWrapper, self).__init__(nn=mlp)
