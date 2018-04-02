import os
import random

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm


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
  
  y_true = np.random.randint(low=0,high=2, size=(64))
  batch_y = Variable(torch.from_numpy(y_true)).cuda()

  return batch_x, batch_y


def read_data(data_file):
  data = []
  with open(data_file, 'r') as f:
    for line in f:
      bug1, bug2 = line.split()
      data.append([int(bug1), int(bug2)])
  return data


train_data = None

def batch_iterator(data, batch_size):
  global train_data
  if not train_data:
    train_data = read_data(os.path.join(data, 'train.txt'))
  random.shuffle(train_data)
  bug_ids = read_bug_ids(data)
  num_batches = int(len(train_data) / batch_size)
  if len(data) % batch_size > 0:
    num_batches += 1
  loop = tqdm(range(num_batches))
  for i in loop:
    batch_bugs = []
    for j in range(batch_size):
      offset = batch_size * i + j
      if offset >= len(train_data):
        break
      neg_bug = get_neg_bug(train_data[offset], bug_ids)
      batch_bugs.append(train_data[offset].append(neg_bug))
    yield loop, read_batch(batch_bugs)

