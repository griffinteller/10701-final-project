import pytest
import torch

from src.ssm import SSMTranslator, SSMTranslatorConfig, ssm, Mamba2Layer, Mamba2Config

def test_ssm_equivalence():
    B_, T_, H_, C_, N_ = 2, 3, 4, 5, 6

    torch.manual_seed(42)
    logA = torch.randn(size=(B_, T_, H_))
    X = torch.randn(size=(B_, T_, H_, C_))
    B = torch.randn(size=(B_, T_, H_, N_))
    C = torch.randn(size=(B_, T_, H_, N_))
    H_n1 = torch.randn(size=(B_, H_, C_, N_))

    Y_linear, H_linear = ssm(logA, X, B, C, H_n1, mode="linear")
    Y_quadratic, H_quadratic = ssm(logA, X, B, C, H_n1, mode="quadratic")

    assert torch.allclose(Y_linear, Y_quadratic, atol=1e-6)
    assert torch.allclose(H_linear, H_quadratic, atol=1e-6)

def test_mamba2layer_runs():
    torch.manual_seed(67)

    config = Mamba2Config(
        conv_filter_size=4,
        model_dim=4,
        num_heads=2,
        state_dim=2
    )

    model = Mamba2Layer(config)

    with pytest.raises(Exception):
        inp = torch.randn(size=(2, 3, 5))
        model(inp)

    inp = torch.randn(size=(2, 3, 4))
    result = model(inp)

    assert inp.shape == result.Y.shape
    assert result.H_T.shape == (
        inp.shape[0], 
        config.num_heads, 
        config.model_dim // config.num_heads, 
        config.state_dim
    )

def test_H_n1_passing_ssm():
    torch.manual_seed(89)

    B_, T_, H_, C_, N_ = 1, 2, 2, 2, 1

    logA = torch.randn(size=(B_, T_, H_))
    X = torch.randn(size=(B_, T_, H_, C_))
    B = torch.randn(size=(B_, T_, H_, N_))
    C = torch.randn(size=(B_, T_, H_, N_))
    H_n1 = torch.randn(size=(B_, H_, C_, N_))

    # print(f"H_n1: {H_n1}")

    Y_T_1, H_1_1 = ssm(logA, X, B, C, H_n1, mode="quadratic")

    Y_0, H_0 = ssm(logA[:, 0:1], X[:, 0:1], B[:, 0:1], C[:, 0:1], H_n1, mode="linear")
    Y_1_2, H_1_2 = ssm(logA[:, 1:2], X[:, 1:2], B[:, 1:2], C[:, 1:2], H_0, mode="linear")

    assert torch.allclose(Y_T_1[:, -1:], Y_1_2, atol=1e-6)
    assert torch.allclose(H_1_1, H_1_2, atol=1e-6)

def test_H_n1_passsing_mamba_layer():
    torch.manual_seed(123)

    B_, T_, H_, C_, N_ = 3, 2, 2, 2, 1

    config = Mamba2Config(
        conv_filter_size=4,
        model_dim=C_ * H_,
        num_heads=H_,
        state_dim=N_
    )

    model = Mamba2Layer(config)

    inp = torch.randn(size=(B_, T_, C_ * H_))
    H_n1 = torch.randn(size=(B_, H_, C_, N_))
    result1 = model(inp, H_n1=H_n1, mode="quadratic")

    partial_result = model(inp[:, 0:1], H_n1=H_n1, mode="linear")
    result2 = model(inp[:, 1:2], H_n1=partial_result.H_T, XBC_cache=partial_result.XBC_cache, mode="linear")

    assert torch.allclose(result1.Y[:, 0], partial_result.Y[:, 0], atol=1e-6)
    assert torch.allclose(result1.Y[:, 1], result2.Y[:, 0], atol=1e-6)
    assert torch.allclose(result1.H_T, result2.H_T, atol=1e-6)

def test_SSM_translator_ag_runs():
    config = SSMTranslatorConfig(
        encoder_n_layers = 3,
        encoder_d_model = 6,
        encoder_n_heads = 2,
        encoder_d_state = 5,
        encoder_vocab_size=3,

        decoder_n_layers = 3,
        decoder_d_model = 6,
        decoder_n_heads = 2,
        decoder_d_state = 3,
        decoder_vocab_size=3,
    )

    translator = SSMTranslator(config)

    ids_in = torch.randint(low=0, high=3, size=(2, 7))

    hs = translator.encode(ids_in)
    for h in hs:
        h.requires_grad_(True)

    logits_out = translator.decode_autoregressive(hs, max_output_len=16)
    # print(logits_out.shape)

    s = logits_out.sum()
    s.backward()

def test_SSM_translator_forced_runs():
    config = SSMTranslatorConfig(
        encoder_n_layers = 3,
        encoder_d_model = 6,
        encoder_n_heads = 2,
        encoder_d_state = 5,
        encoder_vocab_size=3,

        decoder_n_layers = 3,
        decoder_d_model = 6,
        decoder_n_heads = 2,
        decoder_d_state = 3,
        decoder_vocab_size=3,
    )

    translator = SSMTranslator(config)

    ids_in = torch.randint(low=0, high=3, size=(2, 7))
    ids_target = torch.randint(low=0, high=3, size=(2, 17))

    hs = translator.encode(ids_in)
    for h in hs:
        h.requires_grad_(True)

    logits_out = translator.decode_forced(hs, ids_target)
    # print(logits_out.shape)

    s = logits_out.sum()
    s.backward()