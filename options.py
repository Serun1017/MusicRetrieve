import argparse

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")

        # dataset 
        parser.add_argument('--train_sample_data', type=str, default='./test_data/sample')
        parser.add_argument('--train_origin_data', type=str, default='./test_data/origin')
        parser.add_argument('--audio_sampling_rate', type=int, default=16000) # default 16kHz

        # valid set
        parser.add_argument('--valid_sample_data', type=str, default='./test_data/sample')
        parser.add_argument('--valid_origin_data', type=str, default='./test_data/origin')

        # test set
        parser.add_argument('--test_sample_data', type=str, default='./dataset/test/sample')
        parser.add_argument('--test_origin_data', type=str, default='./dataset/test/origin')

        # train
        parser.add_argument('--save', '-s', type=str, default='./checkpoints')
        parser.add_argument('--batch', type=int, default=10)
        parser.add_argument('--epoch', type=int, default=30)
        parser.add_argument('--datasetLen', type=int, default=10000)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=1e-2)

        # test
        parser.add_argument('--load', '-l', type=str, default='./checkpoints/best_checkpoint.pth')
        parser.add_argument('--testall', default=False, action='store_true', help='train/test scale')
        parser.add_argument('--test_sample_batch', type=int, default=20)
        parser.add_argument('--test_origin_batch', type=int, default=20)
        parser.add_argument('--num_workers', type=int, default=4)

        # other
        parser.add_argument('--choose_cuda', '-c', type=str, default='0')
        parser.add_argument("--seed", type=int, default=2021, help="random seed.")


        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
