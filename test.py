import torch

from options import Option
from model.model import Model

if __name__ == '__main__' :
    args = Option().parse()
    model = Model(args).cuda()

    # batch size, channel, data_length
    origin_audio = torch.rand(4, 1, 16000 * 240).cuda()
    sampel_audio = torch.rand(4, 1, 16000 * 10).cuda()

    ca_fea_sample, ca_fea_origin = model(sampel_audio, origin_audio)

    print(ca_fea_sample.size())
    print(ca_fea_sample)

    print(ca_fea_origin.size())
    print(ca_fea_origin)
