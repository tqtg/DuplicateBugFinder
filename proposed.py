import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_emb_matrix


class TextCNN(nn.Module):
  def __init__(self, input_dim, n_filters):
    super(TextCNN, self).__init__()
    self.conv3 = nn.Sequential(
      nn.Conv1d(input_dim, n_filters, kernel_size=3),
      nn.ReLU()
    )
    self.conv4 = nn.Sequential(
      nn.Conv1d(input_dim, n_filters, kernel_size=4),
      nn.ReLU()
    )
    self.conv5 = nn.Sequential(
      nn.Conv1d(input_dim, n_filters, kernel_size=5),
      nn.ReLU()
    )
    self.fc = nn.Sequential(nn.Linear(n_filters * 3, 100), nn.Tanh())

  def forward(self, x_input):
    # Embedding
    x_3 = self.conv3(x_input)
    x_3 = F.relu(F.max_pool1d(x_3, kernel_size=x_3.size()[-1]))
    x_4 = self.conv4(x_input)
    x_4 = F.relu(F.max_pool1d(x_4, kernel_size=x_4.size()[-1]))
    x_5 = self.conv5(x_input)
    x_5 = F.relu(F.max_pool1d(x_5, kernel_size=x_5.size()[-1]))
    x = torch.cat([x_3, x_4, x_5], -2).squeeze()
    return self.fc(x)


class Net(torch.nn.Module):
  def __init__(self, args):
    super(Net, self).__init__()
    self.char_embed = nn.Embedding(args.n_chars, args.char_dim)
    self.word_embed = nn.Embedding(args.n_words, args.word_dim)
    self.word_embed.weight = nn.Parameter(
      torch.from_numpy(load_emb_matrix(args.n_words, args.word_dim, args.data)).float()
    )

    self.word_CNN = TextCNN(args.word_dim, args.n_filters)
    self.char_CNN = TextCNN(args.char_dim, args.n_filters)

    self.word_RNN = nn.GRU(input_size=args.word_dim, hidden_size=50, bidirectional=True, batch_first=True)
    self.char_RNN = nn.GRU(input_size=args.char_dim, hidden_size=50, bidirectional=True, batch_first=True)

    self.info_proj = nn.Sequential(nn.Linear(args.n_prop, 100), nn.Tanh())
    self.feature_proj = nn.Sequential(nn.Linear(500, 500), nn.Tanh(), nn.Linear(500, 500))
    self.projection = nn.Linear(500, 100)

  def forward_cnn(self, x):
    # x = [word, char]
    w_embed = self.word_embed(x[0]).transpose(1, 2)
    c_embed = self.char_embed(x[1]).transpose(1, 2)
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
    info_feature = self.info_proj(x['info'].float())
    word_long, char_long = self.forward_cnn(x['desc'])
    word_short, char_short = self.forward_rnn(x['short_desc'])

    feature = torch.cat([info_feature, word_short, word_long, char_short, char_long], -1)
    feature_residual = F.tanh(feature + self.feature_proj(feature))
    return self.projection(feature_residual)
