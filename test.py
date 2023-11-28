import torch

from options import Option
from model.sa import Scale_Embedding, SelfAttention

if __name__ == '__main__' :
    args = Option().parse()

    origin_audio = torch.zeros(1, 16000 * 240).cuda()
    sampel_audio = torch.zeros(1, 16000 * 10).cuda()
    print(origin_audio.size())

    # scale_embedding = Scale_Embedding(args, 'origin').cuda()
    # convoluded_origin_audio = scale_embedding(origin_audio)
    # print(convoluded_origin_audio.size())

    # scale_embedding = Scale_Embedding(args, 'sample').cuda()
    # cocnvoluded_sample_audio = scale_embedding(sampel_audio)
    # print(cocnvoluded_sample_audio)

    self_attention = SelfAttention(args, type='origin').cuda()
    x = self_attention(origin_audio)

    print(x.size())
    print(x.shape)
    print(x)