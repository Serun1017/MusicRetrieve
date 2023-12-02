import torch
import torch.nn as nn

from einops import rearrange, repeat
import math

# code basic by https://github.com/buptLinfy/ZSE-SBIR. Patch_Embedding has been added and some parameters has been changed
# applying 'Zero-shot Everything Sketch-Based Image Retrieval, and in Explainable Style' paper to music retrieval system by sample of audio

# Scale Embedding of origin or sample audio file. -> [768, 160]
class Patch_Embedding(nn.Module) :
    def __init__(self, args, type='origin') :
        super(Patch_Embedding, self).__init__()

        if type == 'origin' :
            self.seq = nn.Sequential(
                # Input Origin audio file must be 16kHz 4 min [2 * 2 * 10 * 24]. scale embedding to 0.25 sec at each token
                nn.Conv1d(in_channels=2, out_channels=64, kernel_size=args.audio_sampling_rate*10, stride=args.audio_sampling_rate // 4 // 4, padding=int(args.audio_sampling_rate*5) - 1, padding_mode='zeros', bias=False), # [3840] = each token contains 10 sec sample of audio, stride means 0.0625 sec
                nn.ReLU(),
                nn.Conv1d(64, 256, 16, 4, 8, padding_mode='zeros', bias=False), # [960] = each token means 1 sec, stride means 0.25 sec
                nn.ReLU(),
                nn.Conv1d(256, 768, 4, 1, 2, padding_mode='zeros',bias=False), # [960] = each token means 0.25 sec
                nn.ReLU(),
                nn.Conv1d(768, 768, 6, 6, 1, padding_mode='zeros', bias=False) # 960 -> 160 tokens, each token means 0.25 * 6 sec, act like pooling layer
            )
        else :
            self.seq = nn.Sequential(
                # Input Sample audio file must be 16kHz 10 sec
                nn.Conv1d(2, 256, args.audio_sampling_rate*1, args.audio_sampling_rate // 4 // 4, args.audio_sampling_rate // 2, padding_mode='zeros', bias=False), # [160] = each token means 1 sec, stride means 0.0625 sec
                nn.ReLU(),
                nn.Conv1d(256, 768, 4, 1, 1, padding_mode='zeros',bias=False), # [160] = each token means 0.25 sec
            )
    def forward(self, x) :
        return self.seq(x)

class MultiHeadDotProductAttention(nn.Module) :
    def __init__(self, dim, heads, dropout=0.) :
        super(MultiHeadDotProductAttention, self).__init__()

        self.heads = heads
        self.scale = (dim / heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, keep_rate) :
        b, n, c, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        left_tokens = n-1
        if keep_rate < 1 :
            left_tokens = math.cell(keep_rate * left_tokens)
            cls_attn = attn[:, :, 0, 1:] # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1) # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True) # [B, left_tokens]
            idx, _ = torch.sort(idx)
            index = idx.unsqueeze(-1).expand(-1, -1, c) # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens

        return x, None, None, None, left_tokens
    
class FeedForward(nn.Module) :
    def __init__(self, dim, hidden_dim, dropout=0.) :
        super(FeedForward, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x) :
        return self.net(x)

class Encoder1DBlock(nn.Module) :
    def __init__(self, input_shape, heads, mlp_dim, droup_rate=0.1, attention_dropout_rate=0.1) :
        super(Encoder1DBlock, self).__init__()

        self.layer_norm_input = nn.LayerNorm(input_shape)
        self.layer_norm_out = nn.LayerNorm(input_shape)

        self.attention = MultiHeadDotProductAttention(input_shape, heads=heads)
        self.mlp = FeedForward(input_shape, mlp_dim, droup_rate)
        self.drop_out_attention = nn.Dropout(attention_dropout_rate)

    def forward(self, inputs, keep_rate) :
        x = self.layer_norm_input(inputs)
        x, index, idx, cls_attn, left_tokens = self.attention(x, keep_rate)
        x = self.drop_out_attention(x)
        x = x + inputs

        if index is not None :
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)
            x = torch.cat([x[:, 0:1], x_others], dim=1)

        y = self.layer_norm_out(x)
        y = self.mlp(y)

        return x + y, left_tokens, idx

class Encoder(nn.Module) :
    def __init__(self, hidden_size, depth, heads, mlp_dim, droout_rate) :
        super(Encoder, self).__init__()

        self.num_layers = depth
        self.mlp_dim = mlp_dim
        
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList([])
        for _ in range(self.num_layers) :
            self.layers.append(nn.ModuleList([Encoder1DBlock(hidden_size, heads, mlp_dim)]))

        self.keep_rate = (1, ) * 12

    def forward(self, x) :
        left_tokens = []
        idxs = []

        for i, layer in enumerate(self.layers) :
            x, left_token, idx = layer[0](x, self.keep_rate[1])
            left_tokens.append(left_token)
            idxs.append(idx)

        return self.encoder_norm(x), left_tokens, idxs

class SelfAttention(nn.Module) :
    def __init__(self, args, type='origin') :
        super(SelfAttention, self).__init__()

        self.scale_embedding = Patch_Embedding(args, type)
        self.cls = nn.Parameter(torch.randn(1, 1, 768))
        self.pos_embedding = nn.Parameter(torch.randn(1, 160 + 1, 768))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Encoder(768, 12, 12, 3072, 0.1)
       
    def forward(self, audio) :
        x = self.scale_embedding(audio)
        # b: batch, c: channel, w: length of data
        x = rearrange(x, 'b c w ->b w c')
        # reshape the tensor.
        b, n, _ = x.shape


        cls_token = repeat(self.cls, '() n d -> b n d', b=b)
        # add class token to each tokens
        x = torch.cat((cls_token, x), dim=1)
        # position embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        # Transformer Encoder
        sa_fea, left_tokens, idxs = self.transformer(x)

        return sa_fea, left_tokens, idxs
    
