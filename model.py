import torch
from torch import nn


class SlidingAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel: int,
        stride: int = 1,
        num_heads: int = 8,
        out_dim: int = None,
        dropout: float = 0.1
    ):
        super(SlidingAttention, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.sliding_module = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        if out_dim is None:
            out_dim = d_model
        self.linear_module = nn.Linear(d_model, out_dim)

        self.accum_module = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, path):
        seq_len = path.size(-2)
        slices_num = (seq_len - self.kernel) // self.stride

        x = torch.stack([path[i * self.stride:i * self.stride + self.kernel, :] for i in range(slices_num)])

        x, _ = self.sliding_module(x, x, x)

        x = self.linear_module(x)
        x = nn.functional.relu(x)

        cum_att = x[0]
        for i in range(1, x.size(0)):
            cum_att, _ = self.accum_module(x[i], cum_att, cum_att)

        return cum_att


class SlidingTransformer(SlidingAttention):
    def __init__(
        self,
        d_model: int,
        kernel: int,
        stride: int = 1,
        num_heads: int = 8,
        out_dim: int = None,
        dropout: float = 0.1,
    ):
        super(SlidingTransformer, self).__init__(d_model, kernel, stride, num_heads, out_dim, dropout)
        self.sliding_module = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )

        self.accum_module = nn.Transformer(
            d_model=out_dim,
            nhead=num_heads,
            dim_feedforward=4 * out_dim,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, path, is_inverse: bool = True):
        seq_len = path.size(-2)
        slices_num = (seq_len - self.kernel) // self.stride

        x = torch.stack([path[i * self.stride:i * self.stride + self.kernel, :] for i in range(slices_num - 1, -1, -1)])

        x = self.sliding_module(x, x)

        x = self.linear_module(x)
        x = nn.functional.relu(x)

        cum_res = x[0]
        for i in range(1, x.size(0)):
            cum_res = self.accum_module(x[i], cum_res)

        return cum_res


class PollutionPredictor(nn.Module):
    def __init__(self, base_model, out_dim):
        super(PollutionPredictor, self).__init__()
        self.base_model = base_model
        self.out = nn.LazyLinear(out_dim)

    def forward(self, inputs):
        x = self.base_model(inputs)
        x = nn.functional.relu(x)
        x = x.reshape(-1)
        x = self.out(x)
        x = x.reshape(-1, 1)
        return x
