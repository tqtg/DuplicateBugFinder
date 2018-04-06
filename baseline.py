import torch.nn as nn

from cnn import CNN_Text
from data_generator import *
from utils import load_emb_matrix


class BaseNet(torch.nn.Module):
  def __init__(self, args):
    super(BaseNet, self).__init__()
    self.word_embed = nn.Embedding(args.n_words, args.word_dim, max_norm=1, padding_idx=0)
    self.word_embed.weight = nn.Parameter(
      torch.from_numpy(load_emb_matrix(args.n_words, args.word_dim, args.data)).float()
    )
    
    self.CNN = CNN_Text(args.word_dim, args.n_filters)
    self.RNN = nn.GRU(input_size=args.word_dim, hidden_size=50, bidirectional=True, batch_first=True)

    self.info_proj = nn.Sequential(nn.Linear(args.n_prop, 100), nn.Tanh())
    self.projection = nn.Linear(300, 100)


  def forward(self, x):
    # x = [info, desc, short desc]
    info = x['info']
    info_feature = self.info_proj(info.float())

    desc = x['desc'][0]
    desc_feature = self.CNN(self.word_embed(desc))

    short_desc = x['short_desc'][0]
    out, hidden = self.RNN(self.word_embed(short_desc))
    short_desc_feature = torch.mean(out, dim=1)
    
    feature = torch.cat([info_feature, short_desc_feature, desc_feature], -1)
    return self.projection(feature)
