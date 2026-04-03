import torch

from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Literal


@dataclass
class SSMTranslatorConfig:
    encoder_n_layers: int = 4
    encoder_d_model: int  = 1024
    encoder_n_heads: int = 16
    encoder_d_state: int = 64
    encoder_vocab_size: int = 20_000

    decoder_n_layers: int = 4
    decoder_d_model: int  = 1024
    decoder_n_heads: int = 16
    decoder_d_state: int = 64
    decoder_vocab_size: int = 20_000

class SSMTranslator(nn.Module):
    def __init__(self, config: SSMTranslatorConfig):
        super().__init__()

        # for now, only support passing hidden states 1-1
        assert config.decoder_n_layers == config.decoder_d_state

        self.E = nn.Embedding(config.encoder_vocab_size, config.encoder_d_model)
        self.encoder_layers = nn.ModuleList([
            Mamba2Layer(Mamba2Config(
                model_dim=config.encoder_d_model,
                num_heads=config.encoder_n_heads,
                state_dim=config.encoder_d_state,
                disable_output=(i == config.encoder_n_layers - 1)
            )) for i in range(config.encoder_n_layers)
        ])

        encoder_hidden_dim = config.encoder_d_model * config.encoder_d_state
        decoder_hidden_dim = config.decoder_d_model * config.decoder_d_state

        # self.lin_EDs = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        self.lin_EDs = nn.ModuleList([
            nn.Linear(config.encoder_d_state, config.decoder_d_state)
            for i in range(config.encoder_n_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            Mamba2Layer(Mamba2Config(
                model_dim=config.decoder_d_model,
                num_heads=config.decoder_n_heads,
                state_dim=config.decoder_d_state
            )) for _ in range(config.decoder_n_layers)
        ])

        self.D = nn.Linear(config.decoder_d_model, config.decoder_vocab_size)

    def encode(self, ids: torch.Tensor) -> list[torch.Tensor]:
        y = self.E(ids)
        hs = []
        for encoder in self.encoder_layers:
            result: Mamba2LayerResult = encoder(y)
            if result.Y is not None:  # skips last layer
                y = result.Y + y
            hs.append(result.H_T)

        for i, (h, lin) in enumerate(zip(hs, self.lin_EDs)):
            hs[i] = lin(h)

        return hs
    
    def decode_autoregressive(
        self, 
        hs: list[torch.Tensor], 
        max_output_len: int = 1024, 
        pad_id=0,
        bos_id=1,
        eos_id=2,
    ) -> torch.Tensor:
        B, _, _, _ = hs[0].shape

        last_id = torch.full((B,), bos_id)
        logits = []
        XBC_caches: list[None | torch.Tensor] = \
            [None for i in range(len(self.decoder_layers))]
        
        pad_mask = torch.tensor([False for i in range(B)])

        for t in range(max_output_len):
            y = self.E(last_id).unsqueeze(1)  # B x (T = 1) x D

            for i, (h, cache, decoder) in enumerate(zip(hs, XBC_caches, self.decoder_layers)):
                result: Mamba2LayerResult = decoder(
                    inp=y, 
                    H_n1=h, 
                    XBC_cache=cache,
                    mode="linear"
                )

                y = result.Y + y
                hs[i] = result.H_T
                XBC_caches[i] = result.XBC_cache

            lgs = self.D(y).squeeze(1)
            last_id = torch.argmax(lgs, dim=-1)
            last_id[pad_mask] = pad_id
            pad_mask[last_id == eos_id] = True

            logits.append(lgs)

            if pad_mask.all():
                break

        return torch.stack(logits, dim=1)
    
    def decode_forced(
        self,
        hs: list[torch.Tensor],
        inp: torch.Tensor,
    ) -> torch.Tensor:
        
        B = hs[0].shape[0]
        assert inp.shape[0] == B

        logits = []
        XBC_caches: list[None | torch.Tensor] = \
            [None for i in range(len(self.decoder_layers))]
        
        for t in range(inp.shape[1]):
            y = self.E(inp[:, t:t+1])  # B x (T = 1) x D

            for i, (h, cache, decoder) in enumerate(zip(hs, XBC_caches, self.decoder_layers)):
                result: Mamba2LayerResult = decoder(
                    inp=y, 
                    H_n1=h, 
                    XBC_cache=cache,
                    mode="linear"
                )

                y = result.Y + y
                hs[i] = result.H_T
                XBC_caches[i] = result.XBC_cache

            logits.append(self.D(y).squeeze(1))

        return torch.stack(logits, dim=1)


@dataclass
class Mamba2Config:
    conv_filter_size: int = 4
    model_dim: int = 1024
    num_heads: int = 16   # H
    state_dim: int = 64  # N
    disable_output: bool = False

@dataclass 
class Mamba2LayerResult:
    Y: torch.Tensor | None
    H_T: torch.Tensor
    XBC_cache: torch.Tensor


class Mamba2Layer(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()

        assert config.model_dim % config.num_heads == 0

        num_channels = config.model_dim // config.num_heads


        # B = batch, T = length, D = model_dim, H = num_heads, C = num_channels (:= D/H)
        self.config = config

        # B x T x H x C -> B x T x H x 1
        self.lin_logA = MultiheadLinear(num_channels, 1, config.num_heads)

        if config.disable_output:
            channels_XBC = num_channels + config.state_dim
        else:
            channels_XBC = num_channels + 2 * config.state_dim

        # B x T x H x C -> B x T x H x (C + 2N)
        # or B x T x H x (C + N) if no output (no C)
        self.lin_XBC = MultiheadLinear(num_channels, channels_XBC, config.num_heads)

        # B x H(C + 2N) x T -> B x H(C + 2N) x T (depthwise)
        self.conv = nn.Conv1d(
            in_channels=channels_XBC * config.num_heads, 
            out_channels=channels_XBC * config.num_heads,
            kernel_size=config.conv_filter_size,
            groups=channels_XBC * config.num_heads
        )

        if not config.disable_output:
            self.lin_gate = MultiheadLinear(num_channels, num_channels, config.num_heads)

            # B x T x D-> B x T x D
            self.norm = nn.LayerNorm(normalized_shape=[config.model_dim])

            # B x T X D -> B x T x D
            self.lin_out = nn.Linear(config.model_dim, config.model_dim)

    def forward(
        self, 
        inp: torch.Tensor, 
        H_n1: torch.Tensor | None = None,
        XBC_cache: torch.Tensor | None = None,
        mode: Literal["linear", "quadratic"] = "quadratic",
    ) -> Mamba2LayerResult:
        """
        input, output : B x T x D
        """

        # print("---------")

        B_, T_ = inp.shape[:2]
        num_channels = self.config.model_dim // self.config.num_heads

        # B x T x D -> B x T x H x C
        inp = inp.view(size=(B_, T_, self.config.num_heads, num_channels))

        # B x T x H x C -> B x T x H
        logA = self.lin_logA(inp).squeeze(-1)

        # B x H(C + 2N) x T
        XBC_unpadded = self.lin_XBC(inp).flatten(start_dim=-2).permute(0, 2, 1)

        if XBC_cache is None:
            XBC_unconv = F.pad(
                self.lin_XBC(inp).flatten(start_dim=-2).permute(0, 2, 1), 
                (self.config.conv_filter_size - 1, 0),
                "constant",
                0
            )
        else:
            XBC_unconv = torch.concatenate((XBC_cache, XBC_unpadded), dim=-1)

        XBC_cache = XBC_unconv[:, :, -self.config.conv_filter_size:][:, :, 1:]

        XBC_conv = self.conv(XBC_unconv) \
            .permute(0, 2, 1) \
            .reshape(B_, T_, self.config.num_heads, -1)
        
        XBC = F.silu(XBC_conv)
        
        if self.config.disable_output:
            X, B = torch.split(XBC, [num_channels, self.config.state_dim], dim=-1)
            C = None
        else:
            X, B, C = torch.split(XBC, [num_channels, self.config.state_dim, self.config.state_dim], dim=-1)

        # print(f"X: {X}")
        # print(f"B: {B}")
        # print(f"C: {C}")

        if H_n1 is None:
            H_n1 = torch.zeros(
                size=(B_, self.config.num_heads, num_channels, self.config.state_dim),
                device=inp.device
            )

        # print(f"H_n1: {H_n1}")

        Y, H_T = ssm(logA, X, B, C, H_n1, mode=mode, no_Y=self.config.disable_output)

        if self.config.disable_output:
            return Mamba2LayerResult(None, H_T, XBC_cache)

        Y_gated = F.silu(self.lin_gate(inp)) * Y
        Y_gated_flat = torch.flatten(Y_gated, start_dim=-2)
        Y_gated_normed = self.norm(Y_gated_flat)

        res = self.lin_out(Y_gated_normed)
        return Mamba2LayerResult(res, H_T, XBC_cache)


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
    C: torch.Tensor | None,
    H_n1: torch.Tensor,
    mode: Literal["linear", "quadratic", "blocked"] = "quadratic",
    no_Y: bool =False
) -> tuple[torch.Tensor | None, torch.Tensor]:
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
    # print(f"T: {T_}")
    N_ = B.shape[-1]

    assert logA.shape == (B_, T_, H_)
    assert B.shape == (B_, T_, H_, N_)
    assert C is None or C.shape == (B_, T_, H_, N_)

    if mode == "linear":
        # TODO: figure out how to par-scan in linear case

        # B x T x H x C x N
        XB = torch.matmul(X.unsqueeze(-1), B.unsqueeze(-2))

        # B x H x C x N
        H = H_n1

        # B x T x H x 1 x 1
        A = torch.exp(logA).unsqueeze(-1).unsqueeze(-1)

        # B x T x H x C
        if not no_Y:
            Y = torch.empty_like(X)
        else:
            Y = None

        for i in range(T_):
            # print(f"A: ", A[:, i])
            H = H * A[:, i] + XB[:, i]
            # print(f"H_{i}: {H}")
            if not no_Y:
                Y[:, i] = torch.matmul(H, C[:, i].unsqueeze(-1)).squeeze(-1)
                # print(f"Y_{i}: {Y[:, i]}")

        return Y, H
        
    elif mode == "quadratic":
        logA = logA.transpose(-1, -2) # B x H x T
        A_ss = torch.exp(scalar_ss_mat(logA)) # B x H x T x T
        A_0 = torch.exp(logA[:, :, 0]) # B x H

        deltaH = torch.einsum("bhij,bjhn,bjhc->bihcn", A_ss, B, X)
        baseH = torch.einsum("bh,bht,bhcn->bthcn", A_0, A_ss[:, :, :, 0], H_n1)
        H = baseH + deltaH

        if not no_Y:
            # B x T x H x C
            Y = torch.einsum("bthn,bthcn->bthc", C, H)
        else:
            Y = None

        return Y, H[:, -1]
    
    elif mode == "blocked":
        raise NotImplementedError()
    
    else:
        raise ValueError(f"Invalid ssm mode '{mode}'")
    