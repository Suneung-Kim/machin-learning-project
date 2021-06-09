import torch
import torch.nn as nn
import torch.nn.functional as F
import yacs.config

def initialize_weights(module: torch.nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.zeros_(module.bias)


class Fire(nn.Module):

    def __init__(self, inplanes: int, squeeze_planes: int, 
                expand1x1_planes: int, expand3x3_planes: int):
        super().__init__()
        self.inplanes = inplanes
        # self.bn1 = nn.BatchNorm2d(inplanes)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(expand1x1_planes + expand3x3_planes)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(expand3x3_planes)
        
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        x = self.bn1(x)
        x = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
        return self.bn2(x)
    
class Model(nn.Module):

    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()
        input_shape = (1, 1, 36, 60)
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.BatchNorm2d(6),
            Fire(6, 3, 12, 12),
            Fire(24, 3, 12, 12),
            Fire(24, 6, 24, 24),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(48, 6, 24, 24),
            Fire(48, 9, 36, 36),
            Fire(72, 9, 36, 36),
            Fire(72, 12, 48, 48),
            # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 12, 48, 48)
        )
        self.bn = nn.BatchNorm2d(96)
        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=7, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        #     nn.BatchNorm2d(32),
        #     Fire(32, 16, 64, 64),
        #     Fire(128, 16, 64, 64),
        #     Fire(128, 32, 128, 128),
        #     nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        #     Fire(256, 32, 128, 128),
        #     Fire(256, 48, 192, 192),
        #     Fire(384, 48, 192, 192),
        #     Fire(384, 64, 256, 256),
        #     # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        #     Fire(512, 64, 256, 256)
        # )
        # self.bn = nn.BatchNorm2d(512)
        with torch.no_grad():
            self.feature_size = self._forward(
                torch.zeros(*input_shape)).view(-1).size(0)
        self.fc = nn.Linear(self.feature_size + 2, 2)

        self.apply(initialize_weights)

    def _forward(self, x:torch.tensor) -> torch.tensor:
        x = self.features(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x
    def forward(self, x:torch.tensor, y:torch.tensor) -> torch.tensor:
        x = self._forward(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, y], dim=1)
        x = self.fc(x)
        # print('\noutput : ', x, '\n')
        return x