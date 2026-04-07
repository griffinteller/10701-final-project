import torch
from src.transformer import TransformerTranslator, TransformerTranslatorConfig
from src.train import TransformerTranslatorTrainer

def test_transformer_mini_overfit():
    torch.manual_seed(67)
    config = TransformerTranslatorConfig(
        encoder_n_layers=1,
        decoder_n_layers=1,
        encoder_vocab_size=10,
        decoder_vocab_size=10,
        d_model=16,
        n_heads=2,
        ff_dim=32,
        dropout=0.0,
    )

    translator = TransformerTranslator(config)
    model = TransformerTranslatorTrainer(translator)

    inp = torch.tensor([
        [1, 4, 5, 6, 7, 0],
        [1, 6, 7, 8, 0, 0],
        [1, 4, 7, 8, 0, 0],
        [1, 6, 3, 2, 0, 0]
    ])

    target = torch.tensor([
        [1, 7, 8, 9, 0, 0],
        [1, 5, 4, 3, 0, 0],
        [1, 7, 6, 2, 0, 0],
        [1, 5, 6, 2, 0, 0],
    ])

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    model.train()

    for _ in range(200):
        optim.zero_grad()
        loss = model(inp, target)
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.1