# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


onset_net_cfg = {'conv1': (25, 3), 'pool1': (
    3, 2), 'conv2': (7, 3), 'pool2': (3, 1), 'fc1': 1050}


class onsetnet(nn.Module):
    """docstring for onsetnet"""

    def __init__(self):
        super(onsetnet, self).__init__()
        self.config = onset_net_cfg

        self.features = nn.Sequential(
            nn.Conv2d(1, 21, kernel_size=self.config['conv1']),
            nn.BatchNorm2d(21),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.config['pool1'],
                         stride=self.config['pool1']),
            nn.Conv2d(21, 42, kernel_size=self.config['conv2']),
            nn.BatchNorm2d(42),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.config['pool2'],
                         stride=self.config['pool2'])
        )
        self.fc1 = nn.Linear(self.config['fc1'], 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.config['fc1'])
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.sigmoid(x)

