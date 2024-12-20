import torch
import torch.nn as nn
from torch import repeat_interleave
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GATConv, GINConv, GraphNorm, JumpingKnowledge

from discrete_diffusion.modules.mlp import MLP
from discrete_diffusion.utils import get_example_from_batch, get_graph_sizes_from_batch


class NodeEmbedder(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        embedding_dim,
        num_mlp_layers,
        num_convs,
        dropout_rate,
        do_preprocess,
        use_batch_norm=True,
        jump_mode="cat",
        do_time_conditioning=False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_convs = num_convs
        self.jump_mode = jump_mode
        self.do_preprocess = do_preprocess
        self.use_batch_norm = use_batch_norm
        self.do_time_conditioning = do_time_conditioning

        self.preprocess_mlp = (
            MLP(
                num_layers=num_mlp_layers,
                input_dim=self.feature_dim,
                output_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                use_batch_norm=self.use_batch_norm,
            )
            if do_preprocess
            else None
        )

        self.convs = nn.ModuleList()
        for conv in range(self.num_convs):
            input_dim = self.feature_dim if (conv == 0 and not self.do_preprocess) else self.hidden_dim
            output_dim = self.embedding_dim if (conv == self.num_convs - 1) else self.hidden_dim
            # conv = GINConv(
            #     MLP(
            #         num_layers=num_mlp_layers,
            #         input_dim=input_dim,
            #         output_dim=output_dim,
            #         hidden_dim=self.hidden_dim,
            #         use_batch_norm=self.use_batch_norm,
            #     ),
            #     train_eps=True,
            # )
            conv = GATConv(
                in_channels=input_dim,
                out_channels=output_dim,
            )
            self.convs.append(conv)

        if self.do_time_conditioning:
            self.time_mlps = nn.ModuleList()
            for time_mlp in range(self.num_convs):
                output_dim = self.embedding_dim if (time_mlp == self.num_convs - 1) else self.hidden_dim
                time_mlp = MLP(
                    num_layers=num_mlp_layers,
                    input_dim=1,
                    output_dim=output_dim,
                    hidden_dim=self.hidden_dim,
                    use_batch_norm=self.use_batch_norm,
                )
                self.time_mlps.append(time_mlp)

        num_layers = (self.num_convs + 1) if self.do_preprocess else self.num_convs
        pooled_dim = (num_layers * self.hidden_dim) + self.feature_dim if self.jump_mode == "cat" else self.hidden_dim

        self.dropout = nn.Dropout(p=dropout_rate)

        self.mlp = MLP(
            num_layers=num_mlp_layers,
            input_dim=pooled_dim,
            output_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            use_batch_norm=self.use_batch_norm,
        )

        self.jumping_knowledge = JumpingKnowledge(mode=self.jump_mode) if self.jump_mode != "none" else None

    def forward(self, batch, t):
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
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
        node_out_features = self.mlp(h)

        return node_out_features


class NodeEmbedder2(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        embedding_dim,
        num_mlp_layers,
        num_convs,
        dropout_rate,
        do_preprocess,
        use_batch_norm=True,
        jump_mode="cat",
        do_time_conditioning=False,
        num_timesteps=6,
        num_heads=1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_convs = num_convs
        self.jump_mode = jump_mode
        self.do_preprocess = do_preprocess
        self.use_batch_norm = use_batch_norm
        self.do_time_conditioning = do_time_conditioning
        self.num_timesteps = num_timesteps
        self.num_heads = num_heads

        self.GNNs = nn.ModuleList()
        for t in range(self.num_timesteps):
            gnn = nn.ModuleDict()
            preprocess_mlp = (
                MLP(
                    num_layers=num_mlp_layers,
                    input_dim=self.feature_dim,
                    output_dim=self.embedding_dim,
                    hidden_dim=self.hidden_dim,
                    use_batch_norm=self.use_batch_norm,
                )
                if do_preprocess
                else None
            )
            gnn["preprocess_mlp"] = preprocess_mlp

            convs = nn.ModuleList()
            for conv in range(self.num_convs):
                input_dim = self.feature_dim if (conv == 0 and not self.do_preprocess) else self.hidden_dim
                output_dim = self.embedding_dim if (conv == self.num_convs - 1) else self.hidden_dim

                conv = GATConv(in_channels=input_dim, out_channels=output_dim, heads=self.num_heads, concat=False)
                convs.append(conv)
            gnn["convs"] = convs

            num_layers = (self.num_convs + 1) if self.do_preprocess else self.num_convs
            pooled_dim = (
                (num_layers * self.hidden_dim) + self.feature_dim if self.jump_mode == "cat" else self.hidden_dim
            )

            dropout = nn.Dropout(p=dropout_rate)
            gnn["dropout"] = dropout

            mlp = MLP(
                num_layers=num_mlp_layers,
                input_dim=pooled_dim,
                output_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                use_batch_norm=self.use_batch_norm,
            )
            gnn["mlp"] = mlp

            jumping_knowledge = JumpingKnowledge(mode=self.jump_mode) if self.jump_mode != "none" else None
            gnn["jumping_knowledge"] = jumping_knowledge
            self.GNNs.append(gnn)

    def forward(self, batch, timesteps):
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
        :return: embedded graphs, each graph embedded as a point in R^{E}
        """

        # X ~ (num_nodes_in_batch, feature_dim)
        # edge_index ~ (2, num_edges_in_batch)

        t = int(timesteps[0].item())
        X, edge_index = batch.x, batch.edge_index

        if len(X.shape) == 1:
            X = X.unsqueeze(-1)
        gnn = self.GNNs[t - 1]
        h = gnn.preprocess_mlp(X) if self.do_preprocess else X
        jump_xs = [X, h] if self.do_preprocess else [X]

        for conv_step, conv in enumerate(gnn.convs):
            h = conv(h, edge_index)
            jump_xs.append(h)

        if self.jump_mode != "none":
            h = gnn.jumping_knowledge(jump_xs)

        h = gnn.dropout(h)
        # out ~ (num_nodes_in_batch, output_dim)
        node_out_features = gnn.mlp(h)

        return node_out_features
