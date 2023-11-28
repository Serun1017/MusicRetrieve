import torch
import torch.nn as nn

# Scale Embedding of origin or sample audio file. -> [768, 196]
class Scale_Embedding(nn.Module) :
    def __init__(self, args, type='origin') :
        super(Scale_Embedding, self).__init__()

        if type == 'origin' :
            self.seq = nn.Sequential(
                # Input Origin audio file must be 16kHz 4 min [2 * 2 * 10 * 24]. scale embedding to 0.25 sec at each token
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=args.audio_sampling_rate*10, stride=int(args.audio_sampling_rate*0.25*0.25), padding=int(args.audio_sampling_rate*5) - 1, bias=False), # [3840] = each token contains 10 sec sample of audio, stride means 0.0625 sec
                nn.ReLU(),
                nn.Conv1d(64, 256, 16, 4, 7, bias=False), # [960] = each token means 1 sec, stride means 0.25 sec
                nn.ReLU(),
                nn.Conv1d(256, 768, 4, 4, 1,bias=False), # [1 * 16000] = each token means 0.25 sec
                nn.ReLU(),
                nn.Linear(240, 196) # 240 -> 196 tokens
            )
        else :
            self.seq = nn.Sequential(
                # Input Sample audio file must be 16kHz 10 sec
                nn.Conv1d(1, 256, args.audio_sampling_rate*1, int(args.audio_sampling_rate*0.25*0.25), int(args.audio_sampling_rate*0.5), bias=False), # [160] = each token means 1 sec, stride means 0.0625 sec
                nn.ReLU(),
                nn.Conv1d(256, 768, 4, 1, 1, bias=False), # [160] = each token means 0.25 sec
                nn.ReLU(),
                nn.Linear(160, 196) # 160 -> 196
            )
    def forward(self, x) :
        return self.seq(x)

class Encoder(nn.Module) :
    def __init__(self, args) :
        super(Encoder, self).__init__()
    


    def forward(self, x) :

        return x

class SelfAttention(nn.Module) :
    def __init__(self, args, type='origin') :
        super(SelfAttention, self).__init__()

        self.scale_embedding = Scale_Embedding(args, type)
        self.cls = nn.Parameter(torch.randn(768, 1))
        self.pos_embedding = nn.Parameter(torch.randn(1, 196 + 1))
        self.dropout = nn.Dropout(0.1)
       
        self.transformer = Encoder(args)

    def forward(self, audio) :

        x = self.scale_embedding(audio)

        # cat class token to each tokens
        x = torch.cat((self.cls, x), dim=1)
        # position embedding
        x += self.pos_embedding[:, :196 + 1]

        x = self.dropout(x)

        sa_fea, idxs = self.transformer(x)

        return x