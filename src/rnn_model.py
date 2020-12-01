import torch
import torch.nn as nn

class Rnn_Model(nn.Module):
  def __init__(self,size_in=16, size_middle=32, size_out=3, batch_size=10):
    super(Rnn_Model, self).__init__()
    self.hx = torch.zeros(batch_size, size_middle).cuda()
    self.rnn = nn.RNNCell(size_in,size_middle)
    self.sp_fc = nn.Linear(size_middle,size_out)
    self.tp_fc = nn.Linear(size_middle,size_out)
    #self._initialize_weight()

  def forward(self, x):
    self.hx = self.rnn(x,self.hx)
    sp_out = self.sp_fc(self.hx)
    tp_out = self.tp_fc(self.hx)
    return sp_out, tp_out

class Simple_Model(nn.Module):
  def __init__(self,size_in=16, size_middle=32, size_out=3, batch_size=10):
    super(Simple_Model, self).__init__()
    self.hx = torch.zeros(batch_size, size_middle).cuda()
    self.rnn = nn.RNNCell(size_in,size_middle)
    self.fc = nn.Linear(size_middle,size_out)

    #self._initialize_weight()

  def forward(self, x):
    self.hx = self.rnn(x,self.hx)
    out = self.fc(self.hx)
    return out