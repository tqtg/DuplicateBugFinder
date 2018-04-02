import argparse

import torch.optim as optim

import baseline
import data_generator
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data/eclipse')
parser.add_argument('--n_words', type=int, default=50000)
parser.add_argument('--n_chars', type=int, default=100)
parser.add_argument('--word_dim', type=int, default=300)
parser.add_argument('--char_dim', type=int, default=64)
parser.add_argument('--n_filters', type=int, default=128)
parser.add_argument('--n_prop', type=int, default=50)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
args = parser.parse_args()

def train(epoch, net, optimizer):
  print('Epoch: {}'.format(epoch))
  net.train()
  margin = MarginLoss(margin = 1.)
  for batch_idx, (batch_x, batch_x_pos, batch_x_neg) in data_generator.batch_iterator(args.data, args.batch_size):
    optimizer.zero_grad()
    x_features = net(batch_x)
    x_pos_features = net(batch_x_pos)
    x_neg_features = net(batch_x_neg)
    loss = margin(x_features, x_pos_features, x_neg_features)
    loss.backward()
    optimizer.step()

def test(net):
  pass


def main():
  net = baseline.BaseNet(args)
  net.cuda()
  optimizer = optim.Adam(net.parameters(), lr=args.lr)
  for epoch in range(1, args.epochs + 1):
    train(epoch, net, optimizer)
    test(net)


if __name__ == "__main__":
  main()
