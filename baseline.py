import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_emb_matrix


class BaseNet(torch.nn.Module):
  def __init__(self, args):
    super(BaseNet, self).__init__()
    self.word_embed = nn.Embedding(args.n_words, args.word_dim)
    emb_matrix = load_emb_matrix(args.n_words, args.word_dim, args.data)
    self.word_embed.weight = nn.Parameter(torch.from_numpy(emb_matrix))

    self.word_conv3 = nn.Conv1d(args.word_dim, args.n_filters, 3)
    self.word_conv4 = nn.Conv1d(args.word_dim, args.n_filters, 4)
    self.word_conv5 = nn.Conv1d(args.word_dim, args.n_filters, 5)

    self.long_desc_CNN = nn.MaxPool1d(args.n_filters)

    self.short_desc = torch.nn.GRU(input_size =128, hidden_size=256, bidirectional = True)

    self.prop_MLP = nn.Sequential(nn.Linear(args.n_prop, 256), nn.ReLU(),
                                  nn.Linear(256, 128), nn.ReLU())
    self.projection = nn.Linear(args.n_filters * 3 + 128 + 256*2, 128)


  def forward(self, x):
    # x = [info, desc, short desc]
    info = x['info']
    prop_feature = self.prop_MLP(info.float())

    desc = x['desc'][0]
    embedded_desc = self.word_embed(desc).transpose(1, 2)
    w_conv3 = F.relu(self.word_conv3(embedded_desc))
    w_conv3 = F.max_pool1d(w_conv3, kernel_size=w_conv3.size()[-1])
    w_conv4 = F.relu(self.word_conv4(embedded_desc))
    w_conv4 = F.max_pool1d(w_conv4, kernel_size=w_conv4.size()[-1])
    w_conv5 = F.relu(self.word_conv5(embedded_desc))
    w_conv5 = F.max_pool1d(w_conv5, kernel_size=w_conv5.size()[-1])
    long_desc_feature = torch.cat([w_conv3, w_conv4, w_conv5], -2).squeeze()

    short_desc = x['short_desc'][0]
    embedded_short_desc = self.word_embed(short_desc).transpose(1, 2)
    out, hidden = self.short_desc(embedded_short_desc)

    short_desc_feature = torch.mean(out, dim=1)

    feature = torch.cat([prop_feature, short_desc_feature, long_desc_feature], -1)
    feature = F.relu(self.projection(feature))
    pdb.set_trace()
    return feature
