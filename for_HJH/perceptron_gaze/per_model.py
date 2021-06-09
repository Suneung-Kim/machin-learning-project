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
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(124)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(2048)

        self.fc0 = nn.Linear(7,512)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

        # self._initialize_weight()

    # def _initialize_weight(self) -> None:
    #     nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
    #     nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
    def forward(self, x):
        
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # x = self.leakyrelu((self.fc0(x)))
        # x = self.dropout(x)
        # x = self.leakyrelu((self.fc1(x)))
        # x = self.dropout(x)
        # x = self.leakyrelu((self.fc2(x)))
        # x = self.dropout(x)
        # x = self.fc3(x)
        
        
        return x

