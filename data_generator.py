import os
import numpy as np
import torch
from torch.autograd import Variable
import random


def read_bug_ids(data):
  bug_ids = []
  with open(os.path.join(data, 'bug_ids.txt'), 'r') as f:
    for line in f:
      bug_ids.append(int(line.strip()))
  return bug_ids


def get_neg_bug(invalid_bugs, bug_ids):
  neg_bug = random.choice(bug_ids)
  while neg_bug in invalid_bugs:
    neg_bug = random.choice(bug_ids)
  return neg_bug


def read_batch(batch_bugs):
  word1 = Variable(torch.from_numpy(np.random.randint(low=0, high=1999, size=(64, 200)))).cuda()
  char1 = Variable(torch.from_numpy(np.random.randint(low=0, high=99, size=(64, 500)))).cuda()

  word2 = Variable(torch.from_numpy(np.random.randint(low=0, high=1999, size=(64, 50)))).cuda()
  char2 = Variable(torch.from_numpy(np.random.randint(low=0, high=99, size=(64, 200)))).cuda()

  info = Variable(torch.from_numpy(np.random.rand(64, 50))).cuda()

  batch_x = dict()
  batch_x['info'] = info
  batch_x['desc'] = (word1, char1)
  batch_x['short_desc'] = (word2, char2)

  batch_y = Variable(torch.from_numpy(np.random.rand(64, 1))).cuda()

  return batch_x, batch_y


def batch_iterator(data, batch_size):
  train_file = open(os.path.join(data, 'train.txt'), 'r')
  bug_ids = read_bug_ids(data)
  batch_idx = 0
  batch_bugs = []
  for line in train_file:
    bug1, bug2 = line.split()
    neg_bug = get_neg_bug([bug1, bug2], bug_ids)
    batch_bugs.append([bug1, bug2, neg_bug])
    if len(batch_bugs) == batch_size:
      batch_idx += 1
      yield batch_idx, read_batch(batch_bugs)
      batch_bugs = []
  if len(batch_bugs) > 0:  # last batch
    batch_idx += 1
    yield batch_idx, read_batch(batch_bugs)
