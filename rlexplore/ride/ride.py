#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：ride.py
@Author ：Fried
@Date ：2022/9/20 13:47 
'''

from rlexplore.networks.random_encoder import CNNEncoder, MLPEncoder
import numpy as np
import torch



class RIDE:
    def __init__(self,
                 envs,
                 device
                 ):

        if envs.action_space.__class__.__name__ == "Discrete":
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.n
        elif envs.action_space.__class__.__name__ == 'Box':
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.shape
        else:
            raise NotImplementedError
        self.device = device

        if len(self.ob_shape) == 3:
            self.encoder = CNNEncoder(
                kwargs={'in_channels': self.ob_shape[0], 'embedding_size': 128})
        else:
            self.encoder = MLPEncoder(
                kwargs={'input_dim': self.ob_shape[0], 'embedding_size': 64}
            )

        self.encoder.to(self.device)

        for p in self.encoder.parameters():
            p.requires_grad = False



    def compute_irs(self, obs_buffer):
        obs_buffer_tensor = torch.from_numpy(obs_buffer)
        obs_buffer_tensor = obs_buffer_tensor.to(self.device)
        size = obs_buffer_tensor.size()
        intrinsic_rewards = np.zeros(shape=(size[0], size[1]))

        for process in range(size[1]):
            encoded_obs = self.encoder(obs_buffer_tensor[:, process])
            dist = torch.norm(encoded_obs[:-1] - encoded_obs[1:], p=2, dim=1)
            intrinsic_rewards[:-1, process] = dist.cpu().numpy()

        return intrinsic_rewards

