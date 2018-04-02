import numpy as np


def load_vocabulary():
  pass


def load_emb_matrix(vocab_size, emb_size):
  embedding_weights = {}
  f = open('../glove/glove.6B.{}d.txt'.format(emb_size), 'r', encoding='utf-8')
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_weights[word] = coefs
  f.close()
  print('Total {} word vectors in Glove.'.format(len(embedding_weights)))

  embedding_matrix = np.random.uniform(-1.0, 1.0, (vocab_size, emb_size))

  oov_count = 0
  for word, i in load_vocabulary().items():
    embedding_vector = embedding_weights.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
    else:
      oov_count += 1
  print('Number of OOV words: %d' % oov_count)

  return embedding_matrix
