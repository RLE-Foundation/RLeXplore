#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：rnd.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 21:46 
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

class RND(object):
    def __init__(self,
                 envs,
                 device,
                 latent_dim,
                 lr,
                 batch_size,
                 beta,
                 kappa
                 ):
        """
        Exploration by Random Network Distillation (RND)
        Paper: https://arxiv.org/pdf/1810.12894.pdf

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param lr: The learning rate of predictor network.
        :param batch_size: The batch size to train the predictor network.
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
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.kappa = kappa

        if len(self.ob_shape) == 3:
            self.predictor_network = CnnEncoder(
                kwargs={'in_channels': self.ob_shape[0], 'latent_dim': latent_dim})
            self.target_network = CnnEncoder(
                kwargs={'in_channels': self.ob_shape[0], 'latent_dim': latent_dim})
        else:
            self.predictor_network = MlpEncoder(
                kwargs={'input_dim': self.ob_shape[0], 'latent_dim': latent_dim}
            )
            self.predictor_network = MlpEncoder(
                kwargs={'input_dim': self.ob_shape[0], 'latent_dim': latent_dim}
            )

        self.predictor_network.to(self.device)
        self.target_network.to(self.device)

        self.optimizer = optim.Adam(lr=self.lr, params=self.predictor_network.parameters())

        # freeze the network parameters
        for p in self.target_network.parameters():
            p.requires_grad = False

    def compute_irs(self, buffer, time_steps):
        """
        Compute the intrinsic rewards using the collected observations.
        :param buffer: The experiences buffer.
        :param time_steps: The current time steps.
        :return: The intrinsic rewards
        """

        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        intrinsic_rewards = np.zeros_like(buffer.rewards)
        # observations shape ((n_steps, n_envs) + obs_shape)
        n_steps = buffer.observations.shape[0]
        n_envs = buffer.observations.shape[1]
        obs = torch.from_numpy(buffer.observations)
        obs = obs.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                encoded_obs = self.predictor_network(obs[:, idx])
                encoded_obs_target = self.target_network(obs[:, idx])
                dist = torch.norm(encoded_obs - encoded_obs_target, p=2, dim=1)
                dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)

                intrinsic_rewards[:-1, idx] = dist.cpu().numpy()[1:]

        self.update(buffer)

        return beta_t * intrinsic_rewards

    def update(self, buffer):
        n_steps = buffer.observations.shape[0]
        n_envs = buffer.observations.shape[1]
        obs = torch.from_numpy(buffer.observations).reshape(n_steps * n_envs, *self.ob_shape)
        obs = obs.to(self.device)

        dataset = TensorDataset(obs)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            encoded_obs = self.predictor_network(batch_obs)
            encoded_obs_target = self.target_network(batch_obs)

            loss = F.mse_loss(encoded_obs, encoded_obs_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()