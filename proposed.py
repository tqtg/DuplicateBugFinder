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
    self.info_proj = nn.Sequential(nn.Linear(args.n_prop, 100), nn.Tanh())
    self.text_proj = nn.Sequential(nn.Linear(400, 400), nn.Tanh(), nn.Linear(400, 400))
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

    word_short, char_short = self.forward_rnn(short_desc)
    word_long, char_long = self.forward_rnn(desc)
    text_feature = torch.cat([word_short, char_short, word_long, char_long], -1)
    text_residual = text_feature + self.text_proj(text_feature)

    info_feature = self.info_proj(info.float())

    feature = torch.cat([text_residual, info_feature], -1)
    return self.projection(feature)
