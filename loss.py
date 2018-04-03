import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(torch.nn.Module):
  def __init__(self, margin=1.0):
    super(MarginLoss, self).__init__()
    self.margin = margin
    self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

  def forward(self, x, x_pos, x_neg):
    fb1 = self.cos(x, x_pos)
    fb2 = self.cos(x, x_neg)
    loss = self.margin - fb1 + fb2
    loss = F.relu(loss)
    return loss.mean()
