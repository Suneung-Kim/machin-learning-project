import sys
import subprocess
import os
from datetime import datetime
import torchvision
from PIL import Image
import natsort
import numpy as np
import math


# path = 'gaze_data/experiment_2_roll/train/train.txt'
path = 'gaze_data/experiment_2_roll/val/val.txt'
file_open = open(path,'r')

lines_num = len(file_open.readlines())

print(lines_num)
matrix_value = np.zeros((lines_num,7))

file_open = open(path)

idx = 0
for line in file_open.readlines():
    line = line.strip()
    # f = open('gaze_data/experiment_2_roll/train/train_change.txt','a')
    f = open('gaze_data/experiment_2_roll/val/val_change.txt','a')

        # removes all whitespace in string
    line = line.strip()
    # print(line)
    #splits string according to delimiter str
    data = line.split(sep=',')
    # print(data[0])
    print('{},{},{},{},{},{},{},{}'.format(data[0],data[1],data[2],data[4],data[5],data[6],data[7],data[8]),file=f)
    f.close()

    idx += 1

file_open.close()


