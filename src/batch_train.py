import torch
import torch.nn as nn
import inputdata
import numpy as np
import hessianfree
import matplotlib.pyplot as plt
import csv
import rnn_model
from torch.utils.tensorboard import SummaryWriter
  
def test(model,testdata,loss_func,optimizer,sp_test,tp_test):
  sp_right_ans = 0
  tp_right_ans = 0
  sp_total_loss = 0
  tp_total_loss = 0
  #テストデータをスライス
  for i in range(0,testdata.shape[1],10):
    step_input = testdata[:16,i:i+10].T
    #精度の算出
    sp_out, tp_out = model(step_input)
    sp_right = torch.max(sp_out,1)[1].eq(sp_test[i:i+10]).sum().item()
    tp_right = torch.max(tp_out,1)[1].eq(tp_test[i:i+10]).sum().item()
    sp_right_ans += sp_right
    tp_right_ans += tp_right
    #誤差の算出
    sp_loss = loss_func(sp_out,sp_test[i:i+10])
    tp_loss = loss_func(tp_out,tp_test[i:i+10])
    sp_total_loss += sp_loss
    tp_total_loss += tp_loss
  #誤差
  sp_loss = sp_total_loss.data/(list(sp_test.shape)[0]/10)
  tp_loss = tp_total_loss.data/(list(tp_test.shape)[0]/10)
  #精度
  sp_acc = sp_right_ans/list(sp_test.shape)[0]
  tp_acc = tp_right_ans/list(tp_test.shape)[0]
  return sp_acc,tp_acc,sp_loss,tp_loss
  
def train(model,traindata,loss_func,optimizer,sp_train,tp_train):
  optimizer.zero_grad()
  loss = 0
  #学習データをスライス
  for i in range(0,traindata.shape[1],10):
    step_input = traindata[:16,i:i+10].T
    sp_out, tp_out = model(step_input)
    sp_loss = loss_func(sp_out,sp_train[i:i+10])
    tp_loss = loss_func(tp_out,tp_train[i:i+10])
    loss += sp_loss + tp_loss
    loss.backward(retain_graph=True)
    optimizer.step()
    loss = 0

def main(Model, epoch_num):
  epochs = []
  sp_loss_list = []
  tp_loss_list = []
  sp_accuracys = []
  tp_accuracys = []
  model = Model().cuda()
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())
  #optimizer = torch.optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)
  for epoch in range(epoch_num):
    for iteration in range(7):
      #7回毎に変わる学習データ
      traindata, sp_train, tp_train = inputdata.make_train()
      #学習（10入力される毎に誤差逆伝播）
      train(model,traindata,loss_func,optimizer,sp_train,tp_train)
    #評価
    testdata, sp_test, tp_test = inputdata.make_test()
    sp_acc, tp_acc, sp_loss, tp_loss = test(model,testdata, loss_func,optimizer,sp_test, tp_test)
    #表示用
    epoch_str = f'epoch{epoch+1}'
    sp_loss_str = f'sp_loss{sp_loss.data:.4f}'
    tp_loss_str = f'tp_loss{tp_loss.data:.4f}'
    sp_accuracy_str = f'sp_accuracy{sp_acc:.2f}'
    tp_accuracy_str = f'tp_accuracy{tp_acc:.2f}'
    print(f'-----{epoch_str}----')
    print(f'----{sp_accuracy_str} | {sp_loss_str} | {tp_accuracy_str} | {tp_loss_str}----')
    epochs.append(epoch+1)
    sp_loss_list.append(sp_loss.data.item())
    tp_loss_list.append(tp_loss.data.item())
    sp_accuracys.append(sp_acc)
    tp_accuracys.append(tp_acc)
    save_to_data(sp_accuracys, sp_loss_list, tp_accuracys, tp_loss_list)
  return epochs, sp_accuracys, tp_accuracys, sp_loss_list, tp_loss_list

def save_to_data(sp_accuracy, sp_loss, tp_accuracy, tp_loss):
    with open(f'data/allback_sp_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for accuracy1 in sp_accuracy:
            writer.writerow([accuracy1])
    with open(f'data/allback_tp_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for accuracy2 in tp_accuracy:
            writer.writerow([accuracy2])

    with open(f'data/allback_sp_loss.csv', 'w') as f:
        writer = csv.writer(f)
        for loss1 in sp_loss:
            writer.writerow([loss1])
    with open(f'data/allback_tp_loss.csv', 'w') as f:
        writer = csv.writer(f)
        for loss2 in tp_loss:
            writer.writerow([loss2])

def make_image(epoch, sp_accuracy, sp_loss, tp_accuracy, tp_loss):
  #精度のグラフ
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  #plt.ylim(0,1.5)
  plt.plot(epoch, sp_loss, label="spatial")
  plt.plot(epoch, tp_loss, label="tempral")
  plt.legend(loc=0)
  plt.savefig(f'img/adam2_loss.png')
  #誤差のグラフ
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(epoch, sp_accuracy, label="spatial")
  plt.plot(epoch, tp_accuracy, label="tempral")
  #plt.ylim(0,0.7)
  plt.legend(loc=0)
  plt.savefig(f'img/adam2_acc.png')

if __name__ == '__main__':
  epoch, sp_accuracy, tp_accuracy, sp_loss, tp_loss = main(rnn_model.Rnn_Model,200)
  make_image(epoch, sp_accuracy, sp_loss, tp_accuracy, tp_loss)