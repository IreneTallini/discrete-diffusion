import torch.nn as nn
from hydra.utils import instantiate
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GraphNorm, JumpingKnowledge

from discrete_diffusion.modules.mlp import MLP


class LinkPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.node_embedder = instantiate(cfg.node_embedder)

    def forward(self):
        pass
