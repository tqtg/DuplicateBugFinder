import argparse

import numpy as np
import torch.optim as optim

import baseline
import data_generator
import proposed
from data_generator import *
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data/eclipse')
parser.add_argument('--n_words', type=int, default=50000)
parser.add_argument('--n_chars', type=int, default=100)
parser.add_argument('--word_dim', type=int, default=300)
parser.add_argument('--char_dim', type=int, default=64)
parser.add_argument('--n_filters', type=int, default=128)
parser.add_argument('--n_prop', type=int, default=50)
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('-k', '--top_k', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('-b', '--baseline', type=bool, default=False)
args = parser.parse_args()

def train(epoch, net, optimizer):
  print('Epoch: {}'.format(epoch))
  net.train()
  margin = MarginLoss(margin = 0.5)
  for batch_idx, (batch_x, batch_pos, batch_neg) in data_generator.batch_iterator(args.data, args.batch_size):
    pred_pos = net(batch_pos)
    pred_neg = net(batch_neg)
    pred = net(batch_x)
    loss = margin(pred, pred_pos, pred_neg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
      

def export(net, data):
  bug_ids = read_bug_ids(data)
  features = {}

  batch_size = 64
  num_batch = int(len(bug_ids) / batch_size)
  if len(bug_ids) % batch_size > 0:
    num_batch += 1
  loop = tqdm(range(num_batch))
  for i in loop:
    batch_ids = []
    for j in range(batch_size):
      offset = batch_size * i + j
      if offset >= len(bug_ids):
        break
      batch_ids.append(bug_ids[offset])
    batch_features = net(read_batch_bugs(batch_ids, data, test=True))
    for bug_id, feature in zip(batch_ids, batch_features):
      features[bug_id] = feature

  torch.save(features, str(args.net) + '_features.t7')


def test(data, top_k):
  features = torch.load(str(args.net) + '_features.t7')
  test_pairs = read_test_data(os.path.join(data, 'test.txt'))
  cosine_batch = nn.CosineSimilarity(dim=1, eps=1e-6)
  recall = []
  loop = tqdm(range(len(test_pairs)))
  samples_ = torch.stack([features[k] for k in features.keys()])

  for i in loop:
    recall_ = 0.
    idx = test_pairs[i]
    query = idx[0]
    ground_truth = idx[1]

    '''
    samples = random.sample(features.keys(), 100)
    #samples = [x for x in features.keys() if x != query]
    while query in samples:
        samples = random.sample(features.keys(), 100)
    '''

    query_ = features[query].expand(samples_.size(0), 128)
    cos_ = cosine_batch(query_, samples_)

    (_, indices) = torch.topk(cos_, k = top_k)
    candidates = [features.keys()[x.data[0]] for x in indices]

    for m in candidates:
      if m in ground_truth:
        recall_ += 1

    r = float(recall_ / len(ground_truth))
    recall.append(r)
    '''
    true_score = cosine(features[query], features[ground_truth]).data[0]
    if true_score > min(topk):
        recall += 1
    recall /= len(test_pairs)
    '''
  return np.array(recall).mean()


def main():
  if args.baseline:
    net = baseline.BaseNet(args)
    args.net = 'base'
  else:
    net = proposed.Net(args)
    args.net = 'proposed'
  net.cuda()

  if not os.path.exists(str(args.net) + '_checkpoint.t7'):
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
      train(epoch, net, optimizer)
    torch.save(net, str(args.net) + '_checkpoint.t7')
  if not os.path.exists(str(args.net) + '_features.t7'):
    export(net, args.data)
  print('{:.4f}'.format(test(args.data, args.top_k)))
  
if __name__ == "__main__":
  main()
