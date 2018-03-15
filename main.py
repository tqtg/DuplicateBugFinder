import argparse
from baseline import Net
import numpy as np
import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--n_words', type = int, default = 2000)
parser.add_argument('--n_chars', type = int, default = 100)
parser.add_argument('--word_dim', type = int, default = 128)
parser.add_argument('--char_dim', type = int, default = 64)
parser.add_argument('--n_filters', type = int, default = 128)
parser.add_argument('--n_prop', type = int, default = 50)
args = parser.parse_args()


def main():
    net = Net(args)
    net.cuda()
    word1 = Variable(torch.from_numpy(np.random.randint(low= 0, high = 1999, size = (64,200)))).cuda()
    char1 = Variable(torch.from_numpy(np.random.randint(low=0, high= 99, size= (64,500)))).cuda()

    word2 = Variable(torch.from_numpy(np.random.randint(low= 0, high = 1999, size = (64,50)))).cuda()
    char2 = Variable(torch.from_numpy(np.random.randint(low=0, high= 99, size= (64,200)))).cuda()

    info = Variable(torch.from_numpy(np.random.rand(64,50))).cuda()

    x = dict()
    x['short_desc'] = (word2,char2)
    x['long_desc'] = (word1,char1)
    x['info'] = info
    yy = net(x)


if __name__ == "__main__":
    main()