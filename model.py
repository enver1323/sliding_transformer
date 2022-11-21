import torch
from torch import nn, Tensor
import math

from embed import DataEmbedding


class SlidingAttention(nn.Module):
    def __init__(
        self,
        c_in: int,
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

        self.embedding = DataEmbedding(c_in, d_model, dropout)
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
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, path, is_inverse=True):
        path = self.embedding(path)
        seq_len = path.size(-2)
        slices_num = (seq_len - self.kernel) // self.stride + 1

        slices_range = range(slices_num - 1, -1, -1) if is_inverse else range(slices_num)

        cum_res = None

        for i in slices_range:
            x = path[:, i * self.stride:i * self.stride + self.kernel:, ]
            x = self.sliding_module(x, x, x)[0] + x
            x = self.linear_module(x)
            x = nn.functional.relu(x)
            if i == slices_range[0]:
                cum_res = x
            else:
                cum_res = self.accum_module(x, cum_res, cum_res)[0] + x

        return cum_res


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
        self.d_model = d_model
        self.sliding_module = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )

        self.accum_module = nn.Transformer(
            d_model=out_dim,
            nhead=1,
            dim_feedforward=4 * out_dim,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, path, is_inverse: bool = True):
        path = self.embedding(path)
        seq_len = path.size(-2)
        slices_num = (seq_len - self.kernel) // self.stride + 1

        slices_range = range(slices_num - 1, -1, -1) if is_inverse else range(slices_num)

        cum_res = None

        for i in slices_range:
            x = path[:, i * self.stride:i * self.stride + self.kernel:, ]
            x = self.sliding_module(x, x)
            x = self.linear_module(x)
            x = nn.functional.relu(x)
            if i == slices_range[0]:
                cum_res = x
            else:
                cum_res = self.accum_module(x, cum_res)

        return cum_res
