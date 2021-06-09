import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self,x_data,y_label):
        
        self.x_data = x_data

        self.y_label = y_label
        # self.transform = transform
        # self.transform_label = transform_label
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self,idx):
        x = self.x_data[idx]
        # x = x.view(1,-1)
        y = self.y_label[0][idx]

        # x = self.x_data[idx]
        # y = self.y_label[0][idx]

        return x, y
