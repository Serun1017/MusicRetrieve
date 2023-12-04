from utils.valid import valid_cls
from options import Option
from data_utils.dataset import load_data_test
from model.model import Model

if __name__ == "__main__" :
    args = Option().parse()

    model = Model(args).cuda()

    sample, origin = load_data_test(args)
    valid_cls(args, model, sample, origin)