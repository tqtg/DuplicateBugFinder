import argparse
import sys

import torch.optim as optim
import baseline
import data_generator
import proposed
from data_generator import *
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data/eclipse')
parser.add_argument('--n_words', type=int, default=20000)
parser.add_argument('--n_chars', type=int, default=100)
parser.add_argument('--word_dim', type=int, default=300)
parser.add_argument('--char_dim', type=int, default=64)
parser.add_argument('--n_filters', type=int, default=64)
parser.add_argument('--n_prop', type=int, default=651)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('-k', '--top_k', type=int, default=5)
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-b', '--baseline', type=bool, default=False)
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
args = parser.parse_args()


def train(epoch, net, optimizer):
  print('Epoch: {}'.format(epoch))
  net.train()
  losses = []
  margin = MarginLoss(margin=1.0)
  for loop, (batch_x, batch_pos, batch_neg) in data_generator.batch_iterator(args.data, args.batch_size, args.n_neg):
    pred_pos = net(batch_pos)
    pred_neg = net(batch_neg)
    pred = net(batch_x)
    loss = margin(pred, pred_pos, pred_neg)
    losses.append(loss.data[0])
    loop.set_postfix(loss=loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return np.array(losses).mean()


def export(net, data):
  _, bug_ids = read_test_data(data)
  features = {}

  batch_size = 64
  num_batch = int(len(bug_ids) / batch_size)
  if len(bug_ids) % batch_size > 0:
    num_batch += 1
  loop = tqdm(range(num_batch))
  loop.set_description('Exporting')
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
  return features


def test(data, top_k, features=None):
  if not features:
    features = torch.load(str(args.net) + '_features.t7')
  cosine_batch = nn.CosineSimilarity(dim=1, eps=1e-6)
  corrects = 0
  total = 0
  samples_ = torch.stack(features.values())
  test_data, _ = read_test_data(data)
  loop = tqdm(range(len(test_data)))
  loop.set_description('Testing')
  for i in loop:
    idx = test_data[i]
    query = idx[0]
    ground_truth = idx[1]

    query_ = features[query].expand(samples_.size(0), samples_.size(1))
    cos_ = cosine_batch(query_, samples_)

    (_, indices) = torch.topk(cos_, k=top_k + 1)
    candidates = [features.keys()[x.data[0]] for x in indices]

    corrects += len(set(candidates) & set(ground_truth))
    total += len(ground_truth)

  return float(corrects) / total


def main():
  if args.baseline:
    args.net = 'base'
  else:
    args.net = 'proposed'

  if os.path.exists(str(args.net) + '_features.t7'):
    print('Final recall@{}={:.4f}'.format(args.top_k, test(args.data, args.top_k)))
    sys.exit(0)

  if args.net == 'base':
    net = baseline.BaseNet(args)
  else:
    net = proposed.Net(args)
  net.cuda()
  optimizer = optim.Adam(net.parameters(), lr=args.lr)
  best_recall = 0
  best_epoch = 0
  for epoch in range(1, args.epochs + 1):
    loss = train(epoch, net, optimizer)
    features = export(net, args.data)
    recall = test(args.data, args.top_k, features)
    print('Loss={:.4f}, Recall@{}={:.4f}'.format(loss, args.top_k, recall))
    if recall > best_recall:
      best_recall = recall
      best_epoch = epoch
      torch.save(net, str(args.net) + '_checkpoint.t7')
      torch.save(features, str(args.net) + '_features.t7')
  print('Best_epoch={}, Best_recall={:.4f}'.format(best_epoch, best_recall))


if __name__ == "__main__":
  main()
