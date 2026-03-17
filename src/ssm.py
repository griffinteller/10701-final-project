import torch

from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Literal


@dataclass
class Mamba2Config:
    conv_filter_size: int = 4
    model_dim: int = 1024
    num_heads: int = 16   # H
    state_dim: int = 64  # N
    norm_type: Literal["layer"] | Literal["batch"] = "layer"
    ssm_mode: Literal["linear", "quadratic", "blocked"] = "quadratic"
    ssm_block_dim: int = 64

class Mamba2Layer(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()

        assert config.model_dim % config.num_heads == 0

        num_channels = config.model_dim // config.num_heads


        # B = batch, T = length, D = model_dim, H = num_heads, C = num_channels (:= D/H)
        self.config = config

        # B x T x H x C -> B x T x H x 1
        self.lin_logA = MultiheadLinear(num_channels, 1, config.num_heads)

        channels_XBC = num_channels + 2 * config.state_dim

        # B x T x H x C -> B x T x H x (C + 2N)
        self.lin_XBC = MultiheadLinear(num_channels, channels_XBC, config.num_heads)

        self.lin_gate = MultiheadLinear(num_channels, num_channels, config.num_heads)

        # B x T x H(C + 2N) -> B x T x H(C + 2N) (depthwise)
        self.conv = nn.Conv1d(
            in_channels=channels_XBC * config.num_heads, out_channels=channels_XBC * config.num_heads,
            kernel_size=config.conv_filter_size,
            padding=(config.conv_filter_size - 1, 0),
            groups=channels_XBC * config.num_heads
        )

        # B x T x H x C -> B x T x H x C
        self.norm = nn.LayerNorm(normalized_shape=[config.num_heads, num_channels])

        # B x T X D -> B x T x D
        self.lin_out = nn.Linear(config.model_dim, config.model_dim)


    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        input, output : B x T x D
        """

        B, T = inp.shape[:2]
        num_channels = self.config.model_dim // self.config.num_heads

        # B x T x D -> B x T x H x C
        inp = inp.view(size=(B, T, self.config.num_heads, num_channels))

        # B x T x H x C -> B x T x H
        logA = self.lin_logA(inp).squeeze(-1)

        XBC = F.silu(self.lin_XBC(inp))
        X, B, C = torch.split(XBC, [num_channels, self.config.state_dim, self.config.state_dim], dim=-1)

        Y = ssm(logA, X, B, C, mode=self.config.ssm_mode, block_dim=self.config.ssm_block_dim)

        Y_gated = self.lin_gate(inp) * Y
        Y_gated_flat = torch.flatten(Y_gated, start_dim=-2)

        return self.lin_out(Y_gated_flat)


class MultiheadLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(size=(num_heads, out_channels, in_channels)))
        self.bias = nn.Parameter(torch.randn(size=(num_heads, out_channels)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input : B x num_heads x in_channels
        output : B x num_heads x out_channels
        """
        return torch.matmul(self.weights, x.unsqueeze(-1)).squeeze(-1) \
            + self.bias
    

def scalar_ss_mat(
    M: torch.Tensor
) -> torch.Tensor:
    M_scan = torch.cumsum(M, dim=-1)
    M_ss = M_scan[..., :, None] - M_scan[..., None, :]

    triu = torch.triu(
        torch.ones(
            size=(M.shape[-1], M.shape[-1]), 
            device=M.device, 
            dtype=torch.bool), 
        diagonal=1
    )

    M_ss[..., triu] = -torch.inf

    return M_ss


def ssm(
    logA: torch.Tensor, 
    X: torch.Tensor, 
    B: torch.Tensor, 
    C: torch.Tensor,
    mode: Literal["linear", "quadratic", "blocked"] = "quadratic",
    block_dim: int | None = None
):
    """
    inputs
    ------
    A : B x T x H
    X : B x T x H x C
    B : B x T x H x N
    C : B x T x H x N

    output
    ------
    Y : B x T x H x C
    """

    B_, T_, H_, C_ = X.shape
    N_ = B.shape[-1]

    assert logA.shape == (B_, T_, H_)
    assert B.shape == (B_, T_, H_, N_)
    assert C.shape == (B_, T_, H_, N_)

    if mode == "linear":
        # TODO: figure out how to par-scan in linear case

        # B x T x H x C x N
        XB = torch.matmul(X.unsqueeze(-1), B.unsqueeze(-2))

        # B x H x C x N
        H = torch.zeros(size=(B_, H_, C_, N_), device=XB.device, dtype=XB.dtype)

        # B x T x H x 1 x 1
        A = torch.exp(logA).unsqueeze(-1).unsqueeze(-1)

        # B x T x H x C
        Y = torch.empty_like(X)

        for i in range(T_):
            H = H * A[:, i] + XB[:, i]
            Y[:, i] = torch.matmul(H, C[:, i].unsqueeze(-1)).squeeze(-1)

        return Y
        
    elif mode == "quadratic":
        logA = logA.transpose(-1, -2) # B x H x T
        A_ss = torch.exp(scalar_ss_mat(logA)) # B x H x T x T

        return torch.einsum("bhij,bjhn,bihn,bjhc->bihc", A_ss, B, C, X)
    
    elif mode == "blocked":
        raise NotImplementedError()
    
    else:
        raise ValueError(f"Invalid ssm mode '{mode}'")
    