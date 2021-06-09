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
from gaze_train import train, validation
# from models import *
# from utils import progress_bar

def file2matrix(filename, val_col_num, val_col_st_idx, val_col_end_idx, label_idx):

    # ilename: directory and file name 
    # val_col_num: the number of columns which contains numeric values
    # val_col_st_idx: the index of starting column which contains numeric values
    # val_col_end_idx: the index of ending column which contains numeric values 
    # label_idx: the index of label column

    file_open = open(filename,'r')
    lines_num = len(file_open.readlines())

    # print(lines_num)

    # blank matrix and vector to store
    matrix_value = np.zeros((lines_num, val_col_num))
    vector_label = np.zeros((1,lines_num))

    # splits and appends value and label using for loop statemen
    file_open = open(filename)
    idx = 0

    # print(file_open.readlines()[2])
   
    for line in file_open.readlines():
        
        # removes all whitespace in string
        line = line.strip()
        # print(line)
        #splits string according to delimiter str
        list_from_line = line.split(sep=',')

        # #append value to matrix and label to vector
        matrix_value[idx, :] =list_from_line[val_col_st_idx : (val_col_end_idx+1)]
        vector_label[0, idx] = list_from_line[label_idx]
        
        # vector_label.append(list_from_line[label_idx])

        idx += 1
    file_open.close()    
    return matrix_value, vector_label


def one_hot_encode(x):
    output = np.zeros([np.size(x), 2])
    # print(range(np.size(x)))
   
    for i in range(np.size(x)): 
        if x[0,i] == 1:
           output[i,0]=1
        else:
           output[i,1]=1 
    return output     
       


def standardize(numeric_dataset):
    # standardized_value = (x - mean)/ standard_deviation
    # calculate mean and standard deviation per numeric columns

    mean_val = numeric_dataset.mean(axis=0)
    std_dev_val = numeric_dataset.std(axis=0)

    # standardization
    matrix_standardized = (numeric_dataset - mean_val)/ std_dev_val
    
    return matrix_standardized


    


def main():
   best_acc = 0
   device = torch.device('cuda')
   train_value, train_label =file2matrix('./gaze_data/train/train.txt',7,1,7,0)
   val_value, val_label =file2matrix('./gaze_data/validation/val.txt',7,1,7,0)
   

#    train_value = standardize(train_value)
#    val_value = standardize(val_value)


   
   
   train_dataset = CustomDataset(train_value,train_label)
   val_dataset = CustomDataset(val_value,val_label)


   train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle=True, num_workers=2)
   val_dataloader = DataLoader(val_dataset, batch_size = 128, shuffle=True, num_workers=2)
   
   print(train_dataset[0])
   model = Model()
   num_epoch = 3000
   runing_rate = 1e-3

#    for epoch in range(num_epoch):
#        if epoch < 5000:
#            rate = runing_rate
#        elif epoch % 500 == 0:
#            if rate is not 1e-7:
#                rate = rate *0.1
#            else:
#                rate = rate     
#        else:
#            rate = rate
#        train(epoch, train_dataloader, model, device, rate)
#        acc = validation(epoch, val_dataloader, model, device, best_acc)
#        best_acc = acc

    

if __name__ == '__main__':
    main()
