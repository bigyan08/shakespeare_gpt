import torch
import math
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model):
        super(TokenEmbedding,self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)

    def forward(self,x):
        return self.embedding(x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  #(1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        returns the sum of embeddings and positional encoding, hence positional encoded embeddings
        '''
        # shape of x is now: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention,self).__init__()
        assert d_model % num_heads == 0   # d_model must be divisible by num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads


        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def forward(self,query,key,value,mask):
        '''
        shape of query,key,value : batch_size, seq_len, d_model
        '''
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        K = K.view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        V = V.view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)

        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)
        # [B, H, L, d_k] × [B, H, d_k, L] → [B, H, L, L]

        if mask is not None:
            if mask.dim() == 4 and mask.size(-1) == scores.size(-1):
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                raise ValueError(f"Mask shape {mask.shape} not compatible with scores {scores.shape}")


        attn = torch.softmax(scores,dim=-1)
        output = torch.matmul(attn,V)
        #[B, H, L, L] × [B, H, L, d_k] → [B, H, L, d_k]

        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

        return self.W_o(output)



class NormResidual(nn.Module):
    def __init__(self,d_model,dropout=0.1):
        super(NormResidual,self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, sublayer_output):
        return self.layernorm(x + self.dropout(sublayer_output))



class FeedForward(nn.Module):
    def __init__(self, d_model,hidden_layer,dropout=0.1):
        super(FeedForward,self).__init__()
        self.layer1 = nn.Linear(d_model,hidden_layer)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_layer,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.layer2(self.dropout(self.relu(self.layer1(x))))

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,hidden_layer,dropout=0.1):
        super(DecoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention(num_heads,d_model)
        self.res_output1 = NormResidual(d_model,dropout)
        self.ff_output = FeedForward(d_model,hidden_layer,dropout)
        self.res_output2 = NormResidual(d_model,dropout)

    def forward(self,x,mask):
        x = self.res_output1(x, self.self_attn(x,x,x,mask))
        x = self.res_output2(x,self.ff_output(x))
        return x



def build_causal_mask(seq_len, device):
    # Returns shape (1, 1, seq_len, seq_len)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)