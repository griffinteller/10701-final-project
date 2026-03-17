import pytest
import torch

from src.ssm import ssm, Mamba2Layer, Mamba2Config

def test_ssm_equivalence():
    B_, T_, H_, C_, N_ = 2, 3, 4, 5, 6

    torch.manual_seed(42)
    logA = torch.randn(size=(B_, T_, H_))
    X = torch.randn(size=(B_, T_, H_, C_))
    B = torch.randn(size=(B_, T_, H_, N_))
    C = torch.randn(size=(B_, T_, H_, N_))

    Y_linear = ssm(logA, X, B, C, mode="linear")
    Y_quadratic = ssm(logA, X, B, C, mode="quadratic")

    assert torch.allclose(Y_linear, Y_quadratic)

def test_mamba2layer_runs():
    config = Mamba2Config(
        conv_filter_size=4,
        model_dim=4,
        num_heads=2,
        state_dim=2,
        norm_type="layer",
        ssm_mode="quadratic",
    )

    model = Mamba2Layer(config)

    with pytest.raises(Exception):
        inp = torch.randn(size=(2, 3, 5))
        model(inp)

    inp = torch.randn(size=(2, 3, 4))
    out = model(inp)
    assert inp.shape == out.shape