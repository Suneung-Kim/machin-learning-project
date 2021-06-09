import time

import torch
import torchvision.utils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from custom_data import CustomDataset
import torch.optim as optim
import torch.nn as nn
from per_model import Model
import os
def train(epoch, train_dataloader, model, device, runing_rate):
  # print(device)
  c = 1.0
  c = torch.FloatTensor([c]).cuda()
  model.to(device)
  criterion = nn.CrossEntropyLoss()
  # optimizer = torch.optim.SGD(model.parameters(), lr=runing_rate, weight_decay=1e-3)
  # if epoch <=1000:
  #   runing_rate = 1e-6
  # elif epoch >1000 and epoch <=2000:
  #   runing_rate = 1e-7
  # elif epoch >2000 and epoch <=3000:
  #   runing_rate = 1e-8
  # else:
  #   runing_rate = 1e-9

  optimizer = torch.optim.Adam(model.parameters(), lr=runing_rate, weight_decay=1e-4)
  train_loss=0
  total_loss=0
  correct = 0
  for i, (inputs,target) in enumerate(train_dataloader):
    #    print(data)
    #    print(label)
    print(inputs.dim())
    inputs, target = inputs.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, target)
    regularize = 0
    for idx, module in enumerate(model.modules()):
      if isinstance(module, nn.Linear):
        regularize += torch.norm(module.weight.data, p=1)
    regularize.cuda()
    loss = loss + c*regularize
    loss.backward()
    optimizer.step()

    
    train_loss += loss.item()
    # print("@@@@@@@@@@@22",output)
    _, predicted = output.max(1)
    # print("output.max(1)",output.max(1))
    # print(output.max(1))
  
    total_loss += target.size(0)
    correct += predicted.eq(target).sum().item()
           

    print('epoch:', epoch+1, i, len(train_dataloader),'Loss : %.3f | Acc %.3f%%(%d/%d)'
                %(train_loss/(i+1), 100.*correct/total_loss, correct, total_loss))
  

def validation(epoch, val_dataloader, model, device,best_acc):
  criterion = nn.CrossEntropyLoss()
  model.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for i, (inputs, target) in enumerate(val_dataloader):
      inputs, target = inputs.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
      # print("##############3")
      # print(inputs)
      # print("##############")

      output = model(inputs)
      loss = criterion(output,target)

      test_loss += loss.item()
      _, predicted = output.max(1)
      total += target.size(0)
      correct += predicted.eq(target).sum().item()

      print('epoch:', epoch+1, i, len(val_dataloader),'Loss : %.3f | Acc %.3f%%(%d/%d)'
                %(test_loss/(i+1), 100.*correct/total, correct, total))

  ##save
  acc = 100.*correct/total
  if acc > best_acc:
    print('saving..')
    state = {
      'model' : model.state_dict(),
      'acc'   : acc,
      'epoch' : epoch,
    }
    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
    torch.save(model.state_dict(),'./checkpoint/ckpt.pth')
    f=open('checkpoint/check.txt','a')
    print('acc :', acc, 'epoch : ', epoch, file=f)
    f.close()
    best_acc = acc
  return best_acc





