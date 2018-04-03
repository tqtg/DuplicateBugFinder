import argparse

import torch.optim as optim
from tqdm import tqdm

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
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--net', type=str, default='base')
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
      

def export(net, data='../data/eclipse/'):
  bug_ids = read_bug_ids(data)
  feature = {}

  batch_features = net.predict(bug_ids)
  for i, bug_id in enumerate(bug_ids):
    feature[bug_id] = batch_features[i]
 
  torch.save(feature, str(args.net) + '_feature.t7')


def test(data = '../data/eclipse/', threshold = 25):
  features = torch.load(str(args.net) + '_feature.t7')
  test_pairs = read_test_data(os.path.join(data, 'test.txt'))
  cosine_batch = nn.CosineSimilarity(dim=1, eps=1e-6)
  recall = []
  loop = tqdm(range(len(test_pairs)))
  samples_ = torch.stack([features[k] for k in features.keys()])

  for i in loop:
    recall_ = 0.
    idx = test_pairs[i]
    query = idx[0]
    ground_truth = idx[1:]

    '''
    samples = random.sample(features.keys(), 100)
    #samples = [x for x in features.keys() if x != query]
    while query in samples:
        samples = random.sample(features.keys(), 100)
    '''

    query_ = features[query].expand(samples_.size(0), 128)
    cos_ = cosine_batch(query_, samples_)

    (_, indices) = torch.topk(cos_, k = threshold)
    candidates = [features.keys()[x.data[0]] for x in indices]

    for m in candidates:
      if m in ground_truth:
        recall_ += 1
    r = recall_ / len(ground_truth)
    recall.append(r)
    '''
    true_score = cosine(features[query], features[ground_truth]).data[0]
    if true_score > min(topk):
        recall += 1
    recall /= len(test_pairs)
    '''
  return sum(recall)


def main():
  if str(args.net) == 'base':
    net = baseline.BaseNet(args)
  else:
    net = proposed.Net(args)
  net.cuda()
  optimizer = optim.Adam(net.parameters(), lr=args.lr)
  for epoch in range(1, args.epochs + 1):
    train(epoch, net, optimizer)
  
  torch.save(net, str(args.net) + '_checkpoint.t7')
  
  # export(net)
  print(test())
  
if __name__ == "__main__":
  main()
