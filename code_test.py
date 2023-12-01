import torch

from options import Option
from model.model import Model

if __name__ == '__main__' :
    args = Option().parse()
    model = Model(args).cuda()

    # batch size, channel, data_length
    origin_audio = torch.rand(4, 1, 16000 * 240).cuda()
    sampel_audio = torch.rand(4, 1, 16000 * 10).cuda()

    cls_fea, rn_scores = model(sampel_audio, origin_audio)
    
    print(cls_fea.size())
    print(cls_fea)

    print(rn_scores.size())
    print(rn_scores)
