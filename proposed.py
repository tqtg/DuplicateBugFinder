import torch
import torch.nn as nn

from utils import load_emb_matrix


class Net(torch.nn.Module):
  def __init__(self, args):
    super(Net, self).__init__()
    self.word_embed = nn.Embedding(args.n_words, args.word_dim)
    emb_matrix = load_emb_matrix(args.n_words, args.word_dim, args.data)
    self.word_embed.weight = nn.Parameter(torch.from_numpy(emb_matrix).float())
    self.char_embed = nn.Embedding(args.n_chars, args.char_dim)

    self.charRNN = nn.GRU(input_size=args.char_dim, hidden_size=50, bidirectional=True, batch_first=True)
    self.wordRNN = nn.GRU(input_size=args.word_dim, hidden_size=50, bidirectional=True, batch_first=True)

    # self.prop_MLP = nn.Sequential(nn.Linear(args.n_prop, 256), nn.ReLU(),
    #                             nn.Linear(256, 128), nn.ReLU())
    # self.projection = nn.Linear(args.n_filters * 6  + 128, 128)
    self.prop_MLP = nn.Sequential(nn.Linear(args.n_prop, 100), nn.ReLU())
    self.projection = nn.Linear(500, 100)

  def forward_rnn(self, x):
    # x = [word, char]
    word = self.word_embed(x[0])
    char = self.char_embed(x[1])
    out_w, hidden_w = self.wordRNN(word)
    word = torch.mean(out_w, dim=1)

    out_c, hidden_c = self.charRNN(char)
    char = torch.mean(out_c, dim=1)
    return word, char

  def forward(self, x):
    # x = [info, desc, short desc]
    info = x['info']
    desc = x['desc']
    short_desc = x['short_desc']
    long_desc = x['long_desc']

    word_short, char_short = self.forward_rnn(short_desc)
    word_long, char_long = self.forward_rnn(long_desc)
    prop_feature = self.prop_MLP(info.float())
    feat = torch.cat([word_short, char_short, word_long, char_long, prop_feature], -1)
    feature = self.projection(feat)
    return feature
