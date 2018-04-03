import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from tqdm import tqdm
from data_generator import *

class Net(torch.nn.Module):
  def __init__(self, args):
    super(Net, self).__init__()
    self.word_embed = nn.Embedding(args.n_words, args.word_dim)
    self.char_embed = nn.Embedding(args.n_chars, args.char_dim)

    self.word_conv3_short = nn.Conv1d(args.word_dim, args.n_filters, 3, padding = 3)
    self.word_conv4_short = nn.Conv1d(args.word_dim, args.n_filters, 4, padding = 3)
    self.word_conv5_short = nn.Conv1d(args.word_dim, args.n_filters, 5, padding = 3)

    self.char_conv3_short = nn.Conv1d(args.char_dim, args.n_filters, 3, padding = 3)
    self.char_conv4_short = nn.Conv1d(args.char_dim, args.n_filters, 4, padding = 3)
    self.char_conv5_short = nn.Conv1d(args.char_dim, args.n_filters, 5, padding = 3)

    self.word_conv3_long = nn.Conv1d(args.word_dim, args.n_filters, 3, padding = 3)
    self.word_conv4_long = nn.Conv1d(args.word_dim, args.n_filters, 4, padding = 3)
    self.word_conv5_long = nn.Conv1d(args.word_dim, args.n_filters, 5, padding = 3)

    self.char_conv3_long = nn.Conv1d(args.char_dim, args.n_filters, 3, padding = 3)
    self.char_conv4_long = nn.Conv1d(args.char_dim, args.n_filters, 4, padding = 3)
    self.char_conv5_long = nn.Conv1d(args.char_dim, args.n_filters, 5, padding = 3)

    self.prop_MLP = nn.Sequential(nn.Linear(args.n_prop, 256), nn.ReLU(),
                                  nn.Linear(256, 128), nn.ReLU())
    self.projection = nn.Linear(args.n_filters * 6 , 128)
  
  def forward_short(self, x):
    w_emb = self.word_embed(x[0]).transpose(1, 2)
    c_emb = self.char_embed(x[1]).transpose(1, 2)

    w_conv3 = F.relu(self.word_conv3_short(w_emb))
    w_conv3 = F.max_pool1d(w_conv3, kernel_size=w_conv3.size()[-1])
    w_conv4 = F.relu(self.word_conv4_short(w_emb))
    w_conv4 = F.max_pool1d(w_conv4, kernel_size=w_conv4.size()[-1])
    w_conv5 = F.relu(self.word_conv5_short(w_emb))
    w_conv5 = F.max_pool1d(w_conv5, kernel_size=w_conv5.size()[-1])

    c_conv3 = F.relu(self.char_conv3_short(c_emb))
    c_conv3 = F.max_pool1d(c_conv3, kernel_size=c_conv3.size()[-1])
    c_conv4 = F.relu(self.char_conv4_short(c_emb))
    c_conv4 = F.max_pool1d(c_conv4, kernel_size=c_conv4.size()[-1])
    c_conv5 = F.relu(self.char_conv5_short(c_emb))
    c_conv5 = F.max_pool1d(c_conv5, kernel_size=c_conv5.size()[-1])

    w_features = torch.cat([w_conv3, w_conv4, w_conv5], -2).squeeze()
    c_features = torch.cat([c_conv3, c_conv4, c_conv5], -2).squeeze()
    tmp = w_features + c_features
    return tmp

  def forward_long(self, x):
    w_emb = self.word_embed(x[0]).transpose(1, 2)
    c_emb = self.char_embed(x[1]).transpose(1, 2)

    w_conv3 = F.relu(self.word_conv3_long(w_emb))
    w_conv3 = F.max_pool1d(w_conv3, kernel_size=w_conv3.size()[-1])
    w_conv4 = F.relu(self.word_conv4_long(w_emb))
    w_conv4 = F.max_pool1d(w_conv4, kernel_size=w_conv4.size()[-1])
    w_conv5 = F.relu(self.word_conv5_long(w_emb))
    w_conv5 = F.max_pool1d(w_conv5, kernel_size=w_conv5.size()[-1])

    c_conv3 = F.relu(self.char_conv3_long(c_emb))
    c_conv3 = F.max_pool1d(c_conv3, kernel_size=c_conv3.size()[-1])
    c_conv4 = F.relu(self.char_conv4_long(c_emb))
    c_conv4 = F.max_pool1d(c_conv4, kernel_size=c_conv4.size()[-1])
    c_conv5 = F.relu(self.char_conv5_long(c_emb))
    c_conv5 = F.max_pool1d(c_conv5, kernel_size=c_conv5.size()[-1])

    w_features = torch.cat([w_conv3, w_conv4, w_conv5], -2).squeeze()
    c_features = torch.cat([c_conv3, c_conv4, c_conv5], -2).squeeze()
    tmp = w_features + c_features
    return tmp


  def forward(self, x):
    # x = [info, desc, short desc]
    info = x['info']
    desc = x['desc']
    short_desc = x['short_desc']
    short_desc_feature = self.forward_short(short_desc)
    long_desc_feature = self.forward_long(desc)
    #prop_feature = self.prop_MLP(info.float())

    #feature = torch.cat([prop_feature, short_desc_feature, long_desc_feature], -1)
    feature = torch.cat([short_desc_feature, long_desc_feature], -1)
    feature = F.relu(self.projection(feature))
    return feature
  
  def gen(self, bug_ids):
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
      yield loop, read_batch_bugs(batch_ids, data = '../data/eclipse/', test = True)

  def predict(self, bug_ids):
    out = []
    for _, x in self.gen(bug_ids):
        out.extend(self.forward(x))
    return out
