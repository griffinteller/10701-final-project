import torch
import math
from torch import nn
from dataclasses import dataclass
from typing import Literal


@dataclass
class TransformerTranslatorConfig:
    encoder_n_layers: int = 4
    decoder_n_layers: int = 4
    encoder_vocab_size: int = 20_000
    decoder_vocab_size: int = 20_000
    d_model: int = 256
    n_heads: int = 8
    ff_dim: int = 512
    dropout: float = 0.1

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_model, requires_grad=False)

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        self.register_buffer("encoding", encoding)

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]
    
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=1024, drop_prob=0.1):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x) * math.sqrt(self.tok_emb.embedding_dim)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

class Encoder(nn.Module):
    def __init__(self, config: TransformerTranslatorConfig):
        super().__init__()
        
        self.embedding = TransformerEmbedding(
            vocab_size=config.encoder_vocab_size, 
            d_model=config.d_model)
        
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.ff_dim, dropout=config.dropout, batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            layer, num_layers=config.encoder_n_layers)
 
    def forward(self, src, src_key_padding_mask=None):
        x = self.embedding(src)                        
        memory = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory
    
class Decoder(nn.Module):
    def __init__(self, config: TransformerTranslatorConfig):
        super().__init__()
        
        self.embedding = TransformerEmbedding(
            vocab_size=config.decoder_vocab_size, 
            d_model=config.d_model)
        
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.ff_dim, dropout=config.dropout, 
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            layer, num_layers=config.decoder_n_layers)
        
        self.lin_out = nn.Linear(config.d_model, config.decoder_vocab_size)
 
    def forward(self, tgt, memory, tgt_mask=None, 
                memory_key_padding_mask=None, tgt_key_padding_mask=None):
        
        x = self.embedding(tgt)                        
        
        x = self.transformer_decoder(
            x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        
        logits = self.lin_out(x)                        
        return logits
    
class TransformerTranslator(nn.Module):
    def __init__(self, config: TransformerTranslatorConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def decode_forced(self, enc_state, forcing_ids, pad_id: int = 0):
        memory, src_pad_mask = enc_state
        tgt_pad_mask = (forcing_ids == pad_id)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            forcing_ids.size(1), device=forcing_ids.device)
        
        logits = self.decoder(forcing_ids, 
                              memory, 
                              tgt_mask=tgt_mask, 
                              memory_key_padding_mask=src_pad_mask, 
                              tgt_key_padding_mask=tgt_pad_mask)
        return logits
        
    def decode_autoregressive(self, 
                            enc_state, 
                            max_output_len = 1024, 
                            pad_id=0, 
                            bos_id=1, 
                            eos_id=2,):
        memory, src_pad_mask = enc_state
        B = memory.shape[0]

        last_id = torch.full((B,), bos_id, device=memory.device)
        gen_ids = last_id.unsqueeze(1)
        logits_out = []
        pad_mask = torch.tensor([False for i in range(B)], device=memory.device)

        for t in range(max_output_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                gen_ids.size(1), device=gen_ids.device)
            tgt_pad_mask = (gen_ids == pad_id)

            logits = self.decoder(gen_ids, 
                              memory, 
                              tgt_mask=tgt_mask, 
                              memory_key_padding_mask=src_pad_mask, 
                              tgt_key_padding_mask=tgt_pad_mask)
            
            logits = logits[:, -1, :]
            last_id = torch.argmax(logits, dim=-1)

            last_id[pad_mask] = pad_id
            pad_mask[last_id == eos_id] = True

            logits_out.append(logits)

            gen_ids = torch.cat([gen_ids, last_id.unsqueeze(1)], dim=1)

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
        
        src_pad_mask = (inp_ids == pad_id)
        memory = self.encoder(inp_ids, src_key_padding_mask=src_pad_mask)
        enc_state = memory, src_pad_mask

        if decode_method == "ag":
            return self.decode_autoregressive(
                enc_state, 
                max_output_len, 
                pad_id, bos_id, eos_id
            )
        
        elif decode_method == "forced":
            assert forcing_ids is not None
            return self.decode_forced(enc_state, forcing_ids, pad_id)
        
        else:
            raise RuntimeError(f"Invalid decode method '{decode_method}'")
    
