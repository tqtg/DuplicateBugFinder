import argparse
import baseline
import data_generator
from data_generator import *
import torch
import torch.optim as optim
import torch.nn.functional as F
import pdb
from tqdm import tqdm
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data/eclipse')
parser.add_argument('--n_words', type=int, default=50000)
parser.add_argument('--n_chars', type=int, default=100)
parser.add_argument('--word_dim', type=int, default=128)
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
  for batch_idx, (batch_x, batch_y) in data_generator.batch_iterator(args.data, args.batch_size):
    optimizer.zero_grad()
    pred_pos = net(batch_x)
    pred_neg = net(batch_x)
    pred = net(batch_x)
    loss = margin(pred, pred_pos, pred_neg)
    loss.backward()
    optimizer.step()

def export(net, data='../data/eclipse/'):
  bug_ids = read_bug_ids(data)
  feature = {}

  batch_features = net.predict(bug_ids)
  for i, bug_id in enumerate(bug_ids):
    feature[bug_id] = batch_features[i]
 
  torch.save(feature, 'feature.t7')


def test(net, data = '../data/eclipse/', threshold = 5):
  features = torch.load('feature.t7')
  test_pairs = read_data(os.path.join(data, 'test.txt'))[:5]
  cosine = nn.CosineSimilarity(dim = 0, eps= 1e-6)
  recall = 0.
  for idx in test_pairs:
    print(idx)
    query = idx[0]
    match = idx[1]
    top_k = {}
    for k in features.keys():
      if k == query or query not in features.keys():
        continue

      cos = cosine(features[k], features[query])
      top_k[k] = cos.data[0]
    inv_map = {v: k for k, v in top_k.iteritems()}
    topk = sorted(top_k.values())
    topk = topk[-threshold:]
    candidates = [inv_map[x] for x in topk]
    if match in candidates:
        recall += 1
  return recall/len(test_pairs)

def main():
  net = baseline.BaseNet(args)
  net.cuda()
  optimizer = optim.Adam(net.parameters(), lr=args.lr)
  export(net)
  print(test(net))
  '''
  for epoch in range(1, args.epochs + 1):
    train(epoch, net, optimizer)
  torch.save(net.module, 'checkpoint.t7')
  test(net)
  '''
if __name__ == "__main__":
  main()
