from utils.valid import valid_cls
from options import Option
from data_utils.dataset import load_data_test

if __name__ == "__main__" :
    args = Option().parse()

    sample, origin = load_data_test(args)
    valid_cls(args, None, sample, origin)