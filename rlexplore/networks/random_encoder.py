#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：random_encoder.py
@Author ：Fried
@Date ：2022/9/20 13:44 
'''

from torch import nn

class CnnEncoder(nn.Module):
    def __init__(self, kwargs) -> None:
        super(CnnEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(kwargs['in_channels'], 32, (8, 8), stride=(4, 4)), nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2)), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), stride=(1, 1)), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 7 * 7, kwargs['latent_dim']))

    def forward(self, ob):
        x = self.main(ob)

        return x

class MlpEncoder(nn.Module):
    def __init__(self, kwargs) -> None:
        super(MlpEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(kwargs['input_dim'], 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, kwargs['latent_dim'])
        )

    def forward(self, ob):
        x = self.main(ob)

        return x