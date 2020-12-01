import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import math
import random
import itertools

def make_pattern(t_long,patterns):
  sp_patt = []
  tp_patt = []
  ts = torch.arange(0.0, t_long , 1.0)
  cols = [torch.Tensor() for _ in range(16)]
  for p in patterns:
    tp, sp = p
    a = -1
    for i in range(16):
      if i % sp == 0:
        a *= -1
      data = a * torch.cos(2.0*math.pi * tp *ts)
      cols[i] = torch.cat([cols[i], data])
    for j in range(list(ts.shape)[0]):
      #正解データの作成
      if sp == 2:
        spot = 0
        sp_patt.append(spot)
      elif sp == 4:
        spot = 1
        sp_patt.append(spot)
      elif sp == 8:
        spot = 2
        sp_patt.append(spot)
  sp_patt = torch.tensor(sp_patt)
  return cols, sp_patt

def make_seed_patt():
  spatial_patterns = [2, 4, 8]
  temporal_patterns = [1/4, 1/8, 1/16]
  patterns = []
  for sp in spatial_patterns:
    for tp in temporal_patterns:
      patterns.append((tp, sp))
  random.shuffle(patterns)
  return patterns

def make_train(t_long=15):
  patterns = make_seed_patt()
  #学習データの作成
  change_switch = random.randint(0,1)
  if change_switch == 0:
    traindata, train = make_pattern(t_long*2,patterns[0:1])
  else:
    traindata, train = make_pattern(t_long,patterns[0:2])

  traindata = torch.stack(traindata).view(16, -1)
  #学習データをgpuに

  traindata = traindata.cuda()
  train = train.cuda()

  return traindata, train

def make_test(t_long=30):
  patterns = make_seed_patt()

  #シャッフルの後，評価データの作成
  testdata, test = make_pattern(t_long, patterns[0:6])
  testdata = torch.stack(testdata).view(16, -1)
  #評価データをgpuに

  testdata = testdata.cuda()
  test = test.cuda()

  return testdata, test

if __name__ == '__main__':
  traindata, train_ans = make_train()
  testdata, test_ans = make_test()
  print(testdata.shape)
  print(test_ans.shape)
  print(traindata.shape)
  print(train_ans.shape)
  plt.figure()
  plt.imshow(traindata.cpu().numpy())
  plt.savefig(f'img/train_input.png')
  plt.figure()
  plt.imshow(testdata.cpu().numpy())
  plt.savefig(f'img/test_input.png')