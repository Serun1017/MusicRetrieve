import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# code by https://github.com/buptLinfy/ZSE-SBIR

def clones(module, N) :
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, dropout=None, mask=None) :
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None :
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None :
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module) :
    def __init__(self, h, d_model, dropout=0.1) :
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None) :
        if mask is not None :
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for lin, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module) :
    def __init__(self, d_model, d_ff, dropout=0.1) :
        super(PositionwiseFeedForward, self).__init__()
        self.w = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x) :
        return self.w(x)

class LayerNorm(nn.Module) :
    def __init__(self, features, eps=1e-6) :
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x) :
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

class AddAnNorm(nn.Module) :
    def __init__(self, size, dropout) :
        super(AddAnNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y) :
        return self.norm(x + self.dropout(y))

class EncoderLayer(nn.Module) :
    def __init__(self, size, self_attn, feed_forward, dropout) :
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(AddAnNorm(size, dropout), 2)
        self.size = size

    def forward(self, q, k, v, mask) :
        x = self.sublayer[0](v, self.self_attn(q, k, v, mask))
        x = self.sublayer[1](x, self.feed_forward(x))
        return x

class Encoder(nn.Module) :
    def __init__(self, layer, N) :
        super(Encoder, self).__init__()
        self.layer1 = clones(layer, N)
        self.layer2 = clones(layer, N)

    def forward(self, sample, origin, mask) :
        for layer1, layer2 in zip(self.layer1, self.layer2) :
            x_sample = layer1(sample, origin, sample, mask)
            x_origin = layer2(origin, sample, origin, mask)
        return x_sample, x_origin

class CrossAttention(nn.Module) :
    def __init__(self, h=8, n=1, d_model=768, d_ff=1024, dropout=0.1) :
        super(CrossAttention, self).__init__()

        multi_head_attention = MultiHeadAttention(h, d_model)
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoderLayer = EncoderLayer(d_model, multi_head_attention, ffn, dropout)
        
        self.encoder = Encoder(encoderLayer, n)

    def forward(self, sample_sa_fea, origin_sa_fea) :
        x_sampe, x_origin = self.encoder(sample_sa_fea, origin_sa_fea, None)
        return torch.cat((x_sampe, x_origin), dim=0)