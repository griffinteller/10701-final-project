import torch
from src.lstm import LSTMTranslator, LSTMTranslatorConfig
from src.train import LSTMTranslatorTrainer

def test_lstm_mini_overfit():
    torch.manual_seed(67)
    config = LSTMTranslatorConfig(
        encoder_n_layers=1,
        encoder_vocab_size=10,
        decoder_n_layers=1,
        decoder_vocab_size=10,
        embed_dim=16,
        hidden_dim=32,
    )

    translator = LSTMTranslator(config)
    model = LSTMTranslatorTrainer(translator)

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