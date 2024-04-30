import os

from torch import nn
import torch
import torchvision
import importlib
import sys
from torchsummary import summary

import torch
import torch.nn as nn

class Cnn3fc1(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 4 * 6 * 6 * 6, fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc1_nodes, num_classes),
        )

    def forward(self, input):
        x = input
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn3fc2(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 4 * 6 * 6 * 6, fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc1_nodes, fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc2_nodes, num_classes),
        )


    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        x = input
        

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn4fc2(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 8 * 6 * 6 * 6, fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc1_nodes, fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc2_nodes, num_classes),
        )

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        x = input
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn5fc2(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 16 * 6 * 6 * 6, fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc1_nodes, fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc2_nodes, num_classes),
        )

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        x = input
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Cnn6fc2(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 16, base * 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 32),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 32 * 6 * 6 * 6, fc1_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc1_nodes, fc2_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc2_nodes, num_classes),
        )

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        x = input
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Vgg11_3d(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.base = base
        
        self.view_fusion = ['input_concatenation','after_first_conv', 'before_last_conv']
        
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.Conv3d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.Conv3d(base * 16, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),)

        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.ft = nn.Flatten()
        self.dp1 = nn.Dropout()

        nb_fc0 = base * 16 * 6 * 6 * 6

        self.ln1 = nn.Linear(nb_fc0, fc1_nodes)
        self.rl1 = nn.ReLU(inplace=True)
        self.dp1 = nn.Dropout()

        self.ln2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.rl2 = nn.ReLU(inplace=True)
        self.dp2 = nn.Dropout()

        self.ln3 = nn.Linear(fc2_nodes, self.num_classes)


    def _fc_first(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.ft(x)
        x = self.dp1(x)
        return x

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
         
        x = input
        x = self._fc_first(x)
        x = self.ln1(x)
        x = self.rl1(x)
        x = self.dp1(x)

        x = self.ln2(x)
        x = self.rl2(x)
        x = self.dp2(x)

        x = self.ln3(x)

        return x


class Vgg16_3d(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.base = base

        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.Conv3d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.Conv3d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.Conv3d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.Conv3d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.Conv3d(base * 16, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.Conv3d(base * 16, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),)

        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.ft = nn.Flatten()
        self.dp1 = nn.Dropout()

        nb_fc0 = base * 16 * 6 * 6 * 6

        self.ln1 = nn.Linear(nb_fc0, fc1_nodes)
        self.rl1 = nn.ReLU(inplace=True)
        self.dp1 = nn.Dropout()

        self.ln2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.rl2 = nn.ReLU(inplace=True)
        self.dp2 = nn.Dropout()

        self.ln3 = nn.Linear(fc2_nodes, self.num_classes)

    def _fc_first(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.ft(x)
        x = self.dp1(x)
        return x

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        x = input
        x = self._fc_first(x)

        x = self.ln1(x)
        x = self.rl1(x)
        x = self.dp1(x)

        x = self.ln2(x)
        x = self.rl2(x)
        x = self.dp2(x)

        x = self.ln3(x)

        return x


class Vgg19_3d(nn.Module):
    def __init__(self, fc1_nodes=1024, fc2_nodes=1024, num_classes: int = 5, base: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.base = base
        

        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.Conv3d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base),
            nn.MaxPool3d(kernel_size=3, stride=2),  # first down pool

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.Conv3d(base* 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 2),
            nn.MaxPool3d(kernel_size=3, stride=2),  # second down pool

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.Conv3d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.Conv3d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.Conv3d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 4),
            nn.MaxPool3d(kernel_size=3, stride=2),  # third down pool

            nn.Conv3d(base * 4, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 8),
            nn.MaxPool3d(kernel_size=3, stride=2),  # forth down pool

            nn.Conv3d(base * 8, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.Conv3d(base * 16, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.Conv3d(base * 16, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.Conv3d(base * 16, base * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(base * 16),
            nn.MaxPool3d(kernel_size=3, stride=2),)  # fifth down pool

        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.ft = nn.Flatten()
        self.dp1 = nn.Dropout()

        if self.level_node != 0:
            nb_fc0 = base * 16 * 6 * 6 * 6 + 1
        else:
            nb_fc0 = base * 16 * 6 * 6 * 6

        self.ln1 = nn.Linear(nb_fc0, fc1_nodes)
        self.rl1 = nn.ReLU(inplace=True)
        self.dp1 = nn.Dropout()

        self.ln2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.rl2 = nn.ReLU(inplace=True)
        self.dp2 = nn.Dropout()

        self.ln3 = nn.Linear(fc2_nodes, self.num_classes)


    def _fc_first(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.ft(x)
        x = self.dp1(x)
        return x

    def forward(self, input):  # input would be a tuple of size (1,) if only one element is input
        x = input
        x = self._fc_first(x)

        x = self.ln1(x)
        x = self.rl1(x)
        x = self.dp1(x)

        x = self.ln2(x)
        x = self.rl2(x)
        x = self.dp2(x)

        x = self.ln3(x)

        return x



def get_net_3d(name: str,
               nb_cls: int,
               fc1_nodes=1024, 
               fc2_nodes=1024):
   
    if name == 'cnn5fc2':
        net = Cnn5fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls)
    elif name == 'cnn6fc2':
        net = Cnn6fc2(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls)
    elif name == "vgg11_3d":
        net = Vgg11_3d(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls)
    elif name == "vgg16_3d":
        net = Vgg16_3d(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls)
    elif name == "vgg19_3d":
        net = Vgg19_3d(fc1_nodes=fc1_nodes, fc2_nodes=fc2_nodes, num_classes=nb_cls)
    
    else:
        raise Exception('wrong net name', name)

    return net
