import torch
import torch.nn as nn
import simple_input
import numpy as np
import hessianfree
import matplotlib.pyplot as plt
import csv
import rnn_model
from torch.utils.tensorboard import SummaryWriter
  
def test(model,testdata,loss_func,optimizer,test_ans):
  right_ans = 0
  total_loss = 0
  #テストデータをスライス
  for i in range(0,testdata.shape[1],10):
    step_input = testdata[:16,i:i+10].T
    #精度の算出
    out = model(step_input)
    right = torch.max(out,1)[1].eq(test_ans[i:i+10]).sum().item()
    right_ans +=right
    #誤差の算出
    loss = loss_func(out,test_ans[i:i+10])
    total_loss += loss
  #誤差
  loss = total_loss.data/(list(test_ans.shape)[0])
  #精度
  acc = right_ans/(list(test_ans.shape)[0])
  return acc,loss
  
def train(model,traindata,loss_func,optimizer,train_ans):
  out_list = 0
  optimizer.zero_grad()
  loss = 0
  #学習データをスライス
  for i in range(0,traindata.shape[1],10):
    step_input = traindata[:16,i:i+10].T
    out = model(step_input)
    loss = loss_func(out,train_ans[i:i+10])
    #print(loss)
    def closure():
        return loss, out
    loss.backward(retain_graph=True)
    loss, outs=closure()
    optimizer.step(closure)


def main(Model, epoch_num):
  epochs = []
  loss_list = []
  accuracys = []
  model = Model().cuda()
  loss_func = nn.CrossEntropyLoss()
  #optimizer = hessianfree.HessianFree(model.parameters())
  optimizer = torch.optim.Adam(model.parameters())
  for epoch in range(epoch_num):
    for iteration in range(7):
      #7回毎に変わる学習データ
      traindata, train_ans = simple_input.make_train()
      #学習（10入力される毎に誤差逆伝播）
      train(model,traindata,loss_func,optimizer,train_ans)
    #評価
    testdata, test_ans = simple_input.make_test()
    acc, loss = test(model,testdata, loss_func,optimizer,test_ans)
    #表示用
    epoch_str = f'epoch{epoch+1}'
    loss_str = f'loss{loss.data:.4f}'
    accuracy_str = f'accuracy{acc:.2f}'
    print(f'-----{epoch_str}----')
    print(f'----{accuracy_str} | {loss_str} ----')
    epochs.append(epoch+1)
    loss_list.append(loss.data.item())
    accuracys.append(acc)
  return epochs, accuracys, loss_list

def save_to_data(accuracy, loss):
    with open(f'data/allback_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for accuracy1 in accuracy:
            writer.writerow([accuracy1])

    with open(f'data/allback_loss.csv', 'w') as f:
        writer = csv.writer(f)
        for loss1 in loss:
            writer.writerow([loss1])

def make_image(epoch, accuracy, loss):
  #精度のグラフ
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  #plt.ylim(0,1.5)
  plt.plot(epoch, loss, label="spatial")
  plt.legend(loc=0)
  plt.savefig(f'img/hessian_loss.png')
  #誤差のグラフ
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(epoch, accuracy, label="spatial")
  #plt.ylim(0,0.7)
  plt.legend(loc=0)
  plt.savefig(f'img/hessian_acc.png')

if __name__ == '__main__':
  epoch, accuracy, loss = main(rnn_model.Simple_Model,100)
  make_image(epoch, accuracy, loss)
