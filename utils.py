import cPickle as pickle
import os
import sys

import numpy as np


def load_vocabulary(data):
  try:
    with open(os.path.join(data, 'word_vocab.pkl'), 'rb') as f:
      vocab = pickle.load(f)
      print('vocabulary loaded')
      return vocab
  except IOError:
    print('can not load vocabulary')
    sys.exit(0)


def load_emb_matrix(vocab_size, emb_size, data):
  embedding_weights = {}
  f = open('../glove.6B.{}d.txt'.format(emb_size), 'r')
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_weights[word] = coefs
  f.close()
  print('Total {} word vectors in Glove.'.format(len(embedding_weights)))

  embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, emb_size))
  embedding_matrix[0, :] = np.zeros(emb_size)

  oov_count = 0
  vocab = load_vocabulary(data)
  for word, i in vocab.items():
    embedding_vector = embedding_weights.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
    else:
      oov_count += 1
  print('Number of OOV words: %d' % oov_count)

  return embedding_matrix
