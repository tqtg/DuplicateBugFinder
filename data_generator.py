import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm


def train_batch():
  for i in tqdm(range(10)):
    word1 = Variable(torch.from_numpy(np.random.randint(low= 0, high = 1999, size = (64, 200)))).cuda()
    char1 = Variable(torch.from_numpy(np.random.randint(low=0, high= 99, size= (64, 500)))).cuda()

    word2 = Variable(torch.from_numpy(np.random.randint(low= 0, high = 1999, size = (64, 50)))).cuda()
    char2 = Variable(torch.from_numpy(np.random.randint(low=0, high= 99, size= (64, 200)))).cuda()

    info = Variable(torch.from_numpy(np.random.rand(64, 50))).cuda()

    batch_x = dict()
    batch_x['info'] = info
    batch_x['desc'] = (word1, char1)
    batch_x['short_desc'] = (word2, char2)

    batch_y = Variable(torch.from_numpy(np.random.rand(64, 1))).cuda()

    yield i, batch_x, batch_y
