import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Text(nn.Module):
  def __init__(self, input_dim, n_filters):
    super(CNN_Text, self).__init__()
    D = input_dim
    Ci = 1
    Co = n_filters
    Ks = [3, 4, 5]
    self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
    self.fc = nn.Sequential(nn.Linear(n_filters * 3, 100), nn.Tanh())

  def forward(self, x):
    x = x.unsqueeze(1)  # (N, Ci, W, D)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [(N, Co, W), ...]*len(Ks)
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(N, Co), ...]*len(Ks)
    x = torch.cat(x, 1)
    return self.fc(x)