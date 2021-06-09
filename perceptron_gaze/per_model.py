import torch
import torch.nn as nn
import torch.nn.functional as F
import yacs.config


# def initialize_weights(module: torch.nn.Module) -> None:
#     if isinstance(module, nn.Conv2d):
#         nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Linear):
#         nn.init.xavier_uniform_(module.weight)
#         nn.init.zeros_(module.bias)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.6)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.bn0 = nn.BatchNorm1d(32)
        self.bn1 = nn.BatchNorm1d(16)
        # self.bn2 = nn.BatchNorm1d(8)

        self.fc0 = nn.Linear(7,32)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 2)
        # self.fc3 = nn.Linear(8, 2)

        # self._initialize_weight()

    # def _initialize_weight(self) -> None:
    #     nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
    #     nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
    def forward(self, x):
        
        x = self.relu(self.bn0(self.fc0(x)))
        x = self.dropout(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        # x = self.relu(self.bn2(self.fc2(x)))
        # x = self.dropout(x)
        x = self.fc2(x)

        # x = self.leakyrelu((self.fc0(x)))
        # x = self.dropout(x)
        # x = self.leakyrelu((self.fc1(x)))
        # x = self.dropout(x)
        # x = self.leakyrelu((self.fc2(x)))
        # x = self.dropout(x)
        # x = self.fc3(x)
        
        
        return x

