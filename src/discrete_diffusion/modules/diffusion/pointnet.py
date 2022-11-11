import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList


class PointwiseNet(Module):
    def __init__(self, feature_dim, time_dim=3, residual=True):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(feature_dim, 128, time_dim),
            ConcatSquashLinear(128, 256, time_dim),
            ConcatSquashLinear(256, 512, time_dim),
            ConcatSquashLinear(512, 256, time_dim),
            ConcatSquashLinear(256, 128, time_dim),
            ConcatSquashLinear(128, feature_dim, time_dim)
        ])

    def forward(self, x, t):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            t:     Time. (B, ).
        """
        batch_size = x.size(0)
        t = t.view(batch_size, 1, 1)  # (B, 1, 1)

        time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)  # (B, 1, 3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(t=time_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_t):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_t, dim_out, bias=False)
        self._hyper_gate = Linear(dim_t, dim_out)

    def forward(self, t, x):
        gate = torch.sigmoid(self._hyper_gate(t))
        bias = self._hyper_bias(t)
        ret = self._layer(x) * gate + bias
        return ret
