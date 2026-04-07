import torch
from torch import nn
from dataclasses import dataclass
from typing import Literal

@dataclass
class LSTMTranslatorConfig:
    encoder_n_layers: int = 4
    encoder_vocab_size: int = 20_000
    decoder_n_layers: int = 4
    decoder_vocab_size: int = 20_000
    embed_dim: int = 256
    hidden_dim: int = 512

class Encoder(nn.Module):
    def __init__(self, config: LSTMTranslatorConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.encoder_vocab_size, config.embed_dim)
        self.LSTM = nn.LSTM(
            config.embed_dim, 
            config.hidden_dim, 
            config.encoder_n_layers,
            batch_first=True)
        
    def forward(self, x):
        input = self.embedding(x)
        _, (hidden, cell) = self.LSTM(input)
        return hidden, cell
        
class Decoder(nn.Module):
    def __init__(self, config: LSTMTranslatorConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.decoder_vocab_size, config.embed_dim)
        self.LSTM = nn.LSTM(
            config.embed_dim, 
            config.hidden_dim,
            config.decoder_n_layers,
            batch_first=True)
        self.lin_out = nn.Linear(config.hidden_dim, config.decoder_vocab_size)

    def forward(self, x, hidden, cell):
        emb = self.embedding(x)
        outputs, (hidden, cell) = self.LSTM(emb, (hidden, cell))
        logits = self.lin_out(outputs)
        return logits, (hidden, cell)

class LSTMTranslator(nn.Module):
    def __init__(self, config:LSTMTranslatorConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def decode_forced(self, hs, inp):
        hidden, cell = hs
        logits, _ = self.decoder(inp, hidden, cell)
        return logits
        
    def decode_autoregressive(self, 
                            hs, 
                            max_output_len = 1024, 
                            pad_id=0, 
                            bos_id=1, 
                            eos_id=2):
        hidden, cell = hs
        B = hidden.shape[1]

        last_id = torch.full((B,), bos_id)
        logits_out = []
        pad_mask = torch.tensor([False for i in range(B)])

        for t in range(max_output_len):
            cur_ids = last_id.unsqueeze(1)
            
            logits, (hidden, cell) = self.decoder(cur_ids, hidden, cell)
            logits = logits.squeeze(1)
            
            last_id = torch.argmax(logits, dim=-1)
            last_id[pad_mask] = pad_id
            
            pad_mask[last_id == eos_id] = True
            logits_out.append(logits)

            if pad_mask.all():
                break

        return torch.stack(logits_out, dim=1)

    def forward(self, 
                inp_ids, 
                decode_method: Literal["ag", "forced"]="ag", 
                forcing_ids: torch.Tensor | None = None,
                max_output_len: int = 1024,
                pad_id: int = 0,
                bos_id: int = 1,
                eos_id: int = 2):
        
        hs = self.encoder(inp_ids)
    
        if decode_method == "ag":
            return self.decode_autoregressive(
                hs, 
                max_output_len, 
                pad_id, bos_id, eos_id
            )
        
        elif decode_method == "forced":
            assert forcing_ids is not None
            return self.decode_forced(hs, forcing_ids)
        
        else:
            raise RuntimeError(f"Invalid decode method '{decode_method}'")


