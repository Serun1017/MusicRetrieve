import torch

from options import Option
from model.sa import Patch_Embedding, SelfAttention

if __name__ == '__main__' :
    args = Option().parse()

    origin_audio = torch.zeros(1, 1, 16000 * 240).cuda()
    sampel_audio = torch.zeros(1, 1, 16000 * 10).cuda()

    scale_embedding = Patch_Embedding(args, 'origin').cuda()
    convoluded_origin_audio = scale_embedding(origin_audio)

    scale_embedding = Patch_Embedding(args, 'sample').cuda()
    cocnvoluded_sample_audio = scale_embedding(sampel_audio)

    print(convoluded_origin_audio.size())
    print(cocnvoluded_sample_audio.size())

    self_attention_origin = SelfAttention(args, type='origin').cuda()
    sa_fea, left_tokens, idxs = self_attention_origin(origin_audio)
    print(sa_fea, left_tokens, idxs)

    self_attention_sample = SelfAttention(args, type='sample').cuda()
    sa_fea, left_tokens, idxs = self_attention_sample(sampel_audio)
    print(sa_fea, left_tokens, idxs)