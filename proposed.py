import torch
import torch.nn as nn

from cnn import CNN_Text
from utils import load_emb_matrix


class Residual(nn.Module):
  def __init__(self, d, fn):
    super(Residual, self).__init__()
    self.fn = fn
    self.projection = nn.Sequential(nn.Linear(d, d), fn, nn.Linear(d, d))

  def forward(self, x):
    return self.fn(x + self.projection(x))


class Net(nn.Module):
  def __init__(self, args):
    super(Net, self).__init__()
    self.char_embed = nn.Embedding(args.n_chars, args.char_dim, max_norm=1, padding_idx=0)
    self.word_embed = nn.Embedding(args.n_words, args.word_dim, max_norm=1, padding_idx=0)
    self.word_embed.weight = nn.Parameter(
      torch.from_numpy(load_emb_matrix(args.n_words, args.word_dim, args.data)).float()
    )

    self.word_CNN = CNN_Text(args.word_dim, args.n_filters)
    self.char_CNN = CNN_Text(args.char_dim, args.n_filters)

    self.word_RNN = nn.GRU(input_size=args.word_dim, hidden_size=50, bidirectional=True, batch_first=True)
    self.char_RNN = nn.GRU(input_size=args.char_dim, hidden_size=50, bidirectional=True, batch_first=True)

    self.info_proj = nn.Sequential(nn.Linear(args.n_prop, 100), nn.Tanh())
    self.residual = Residual(500, nn.Tanh())
    self.projection = nn.Linear(500, 100)

  def forward_cnn(self, x):
    # x = [word, char]
    w_embed = self.word_embed(x[0])
    c_embed = self.char_embed(x[1])
    return self.word_CNN(w_embed), self.char_CNN(c_embed)

  def forward_rnn(self, x):
    # x = [word, char]
    out_w, _ = self.word_RNN(self.word_embed(x[0]))
    out_w = torch.mean(out_w, dim=1)
    out_c, _ = self.char_RNN(self.char_embed(x[1]))
    out_c = torch.mean(out_c, dim=1)
    return out_w, out_c

  def forward(self, x):
    # x = [info, desc, short desc]
    info = x['info']
    info_feature = self.info_proj(info.float())
    word_long, char_long = self.forward_cnn(x['desc'])
    word_short, char_short = self.forward_rnn(x['short_desc'])

    # feature = torch.cat([info_feature, word_short, word_long], -1)
    feature = torch.cat([info_feature, word_short, word_long, char_short, char_long], -1)
    feature_res = self.residual(feature)
    return self.projection(feature_res)
