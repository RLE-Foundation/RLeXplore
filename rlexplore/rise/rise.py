#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：rise.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 13:38 
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder
import torch
import numpy as np

class RISE(object):
    def __init__(self,
                 envs,
                 device,
                 latent_dim,
                 beta,
                 kappa
                 ):
        """
        Rényi State Entropy Maximization for Exploration Acceleration in Reinforcement Learning (RISE)
        Paper: https://ieeexplore.ieee.org/abstract/document/9802917/

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """

        if envs.action_space.__class__.__name__ == "Discrete":
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.n
        elif envs.action_space.__class__.__name__ == 'Box':
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.shape
        else:
            raise NotImplementedError
        self.device = device
        self.beta = beta
        self.kappa = kappa

        if len(self.ob_shape) == 3:
            self.encoder = CnnEncoder(
                kwargs={'in_channels': self.ob_shape[0], 'latent_dim': latent_dim})
        else:
            self.encoder = MlpEncoder(
                kwargs={'input_dim': self.ob_shape[0], 'latent_dim': latent_dim}
            )

        self.encoder.to(self.device)

        # freeze the network parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

    def compute_irs(self, buffer, time_steps, alpha=0.5, k=3):
        """
        Compute the intrinsic rewards using the collected observations.
        :param buffer: The experiences buffer.
        :param time_steps: The current time steps.
        :param alpha: The order of Rényi divergence.
        :param k: The k value.
        :return: The intrinsic rewards
        """

        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        # observations shape ((n_steps, n_envs) + obs_shape)
        obs_tensor = torch.from_numpy(buffer.observations)
        obs_tensor = obs_tensor.to(self.device)
        size = obs_tensor.size()
        intrinsic_rewards = np.zeros(shape=(size[0], size[1]))

        for process in range(size[1]):
            encoded_obs = self.encoder(obs_tensor[:, process])
            for step in range(size[0]):
                dist = torch.norm(encoded_obs[step] - encoded_obs, p=2, dim=1)
                D_step = torch.pow(torch.kthvalue(dist, k + 1).values, 1. - alpha)
                intrinsic_rewards[step, process] = D_step

        return beta_t * intrinsic_rewards