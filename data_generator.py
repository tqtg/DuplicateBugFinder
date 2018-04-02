import cPickle as pickle
import os
import random
import pdb
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


def data_padding(data):
  max_seq_size = max([len(seq) for seq in data])
  padded_data = np.zeros(shape=[len(data), max_seq_size])
  for i, seq in enumerate(data):
    for j, token in enumerate(seq):
      padded_data[i, j] = int(token)
  return padded_data.astype(np.int)


def read_batch_bugs(batch_bugs, data, test = False):
  desc_word = []
  desc_char = []
  short_desc_word = []
  short_desc_char = []
  for bug_id in batch_bugs:
    bug = pickle.load(open(os.path.join(data, 'bugs', '{}.pkl'.format(20)), 'rb'))
    desc_word.append(bug['description_word'])
    desc_char.append(bug['description_char'])
    short_desc_word.append(bug['short_description_word'])
    short_desc_char.append(bug['short_description_char'])
  sz = len(desc_word)
  desc_word = Variable(torch.from_numpy(data_padding(desc_word))).cuda()
  desc_char = Variable(torch.from_numpy(data_padding(desc_char))).cuda()

  short_desc_word = Variable(torch.from_numpy(data_padding(short_desc_word)), volatile = test).cuda()
  short_desc_char = Variable(torch.from_numpy(data_padding(short_desc_char)), volatile = test).cuda()

  info = Variable(torch.from_numpy(np.random.rand(sz, 50)), volatile = test).cuda()

  batch_bugs = dict()
  batch_bugs['info'] = info
  batch_bugs['desc'] = (desc_word, desc_char)
  batch_bugs['short_desc'] = (short_desc_word, short_desc_char)

  return batch_bugs


def read_batch_triplets(batch_triplets, data):
  batch_input_bugs = []
  batch_pos_bugs = []
  batch_neg_bugs = []
  for triplet in batch_triplets:
    batch_input_bugs.append(triplet[0])
    batch_pos_bugs.append(triplet[1])
    batch_neg_bugs.append(triplet[2])
  return read_batch_bugs(batch_input_bugs, data), \
         read_batch_bugs(batch_pos_bugs, data), \
         read_batch_bugs(batch_neg_bugs, data)


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
    batch_triplets = []
    for j in range(batch_size):
      offset = batch_size * i + j
      if offset >= len(train_data):
        break
      neg_bug = get_neg_bug(train_data[offset], bug_ids)
      train_data[offset].append(neg_bug)
      batch_triplets.append(train_data[offset])
    yield loop, read_batch_triplets(batch_triplets, data)
