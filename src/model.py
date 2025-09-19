import torch.nn as nn
import src.config as cfg
from src.layers import TokenEmbedding,PositionalEncoding,DecoderLayer,build_causal_mask


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        assert cfg.VOCAB_SIZE is not None
        self.token_emb = TokenEmbedding(cfg.VOCAB_SIZE,cfg.D_MODEL)
        self.pos_enc = PositionalEncoding(cfg.D_MODEL,cfg.BLOCK_SIZE) 
        self.decoder = nn.ModuleList([
            DecoderLayer(cfg.D_MODEL, cfg.NUM_HEADS, cfg.HIDDEN_LAYER, cfg.DROPOUT)
            for _ in range(cfg.NUM_DEC)
        ])
        self.output_layer = nn.Linear(cfg.D_MODEL,cfg.VOCAB_SIZE)

    def forward(self,idx):
        B,L = idx.shape
        x = self.token_emb(idx)
        x = self.pos_enc(x)
        mask = build_causal_mask(L,cfg.DEVICE)
        for layer in self.decoder:
            x = layer(x,mask)
        output = self.output_layer(x)
        return output