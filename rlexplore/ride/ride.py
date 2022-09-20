#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：ride.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 13:47 
'''

from rlexplore.networks.inverse_forward_networks import InverseForwardDynamicsModel, CnnEncoder

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np



class RIDE:
    def __init__(self,
                 envs,
                 device,
                 lr,
                 batch_size,
                 beta,
                 kappa
                 ):
        """
        RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments
        Paper: https://arxiv.org/pdf/2002.12292

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param lr: The learning rate of inverse and forward dynamics model.
        :param batch_size: The batch size to train the dynamics model.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """
        self.device = device
        self.beta = beta
        self.kappa = kappa
        self.lr = lr
        self.batch_size = batch_size

        if envs.action_space.__class__.__name__ == "Discrete":
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.n
            self.action_type = 'dis'
            self.inverse_forward_model = InverseForwardDynamicsModel(
                kwargs={'latent_dim': 1024, 'action_dim': self.action_shape}
            ).to(device)
            self.im_loss = nn.CrossEntropyLoss()
        elif envs.action_space.__class__.__name__ == 'Box':
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.shape
            self.action_type = 'cont'
            self.inverse_forward_model = InverseForwardDynamicsModel(
                kwargs={'latent_dim': self.ob_shape[0], 'action_dim': self.action_shape[0]}
            ).to(device)
            self.im_loss = nn.MSELoss()
        else:
            raise NotImplementedError
        self.fm_loss = nn.MSELoss()

        if len(self.ob_shape) == 3:
            self.cnn_encoder = CnnEncoder(kwargs={'in_channels': 4}).to(device)

        self.optimizer = optim.Adam(lr=self.lr, params=self.inverse_forward_model.parameters())

    def update(self, buffer):
        n_steps = buffer.observations.shape[0]
        n_envs = buffer.observations.shape[1]
        obs = torch.from_numpy(buffer.observations).reshape(n_steps * n_envs, *self.ob_shape)
        if self.action_type == 'dis':
            actions = torch.from_numpy(buffer.actions).reshape(n_steps * n_envs, )
            actions = F.one_hot(actions.to(torch.int64), self.action_shape).float()
        else:
            actions = torch.from_numpy(buffer.actions).reshape(n_steps * n_envs, self.action_shape[0])
        obs = obs.to(self.device)
        actions = actions.to(self.device)

        if len(self.ob_shape) == 3:
            encoded_obs = self.cnn_encoder(obs)
        else:
            encoded_obs = obs

        dataset = TensorDataset(encoded_obs[:n_steps - 1], actions[:n_steps - 1], encoded_obs[1:n_steps])
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            batch_actions = batch_data[1]
            batch_next_obs = batch_data[2]

            pred_actions, pred_next_obs = self.inverse_forward_model(
                batch_obs, batch_actions, batch_next_obs
            )

            loss = self.im_loss(pred_actions, batch_actions) + \
                   self.fm_loss(pred_next_obs, batch_next_obs)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def pseudo_count(self,
                     encoded_obs,
                     k=10,
                     kernel_cluster_distance=0.008,
                     kernel_epsilon=0.0001,
                     c=0.001,
                     sm=8,
                     ):
        counts = np.zeros(shape=(encoded_obs.size()[0], ))
        for step in range(encoded_obs.size(0)):
            ob_dist = torch.norm(encoded_obs[step] - encoded_obs, p=2, dim=1)
            ob_dist = torch.sort(ob_dist).values
            ob_dist = ob_dist[:k]
            dist = ob_dist.cpu().numpy()
            # TODO: moving average
            dist = dist / np.mean(dist)
            dist = np.max(dist - kernel_cluster_distance, 0)
            kernel = kernel_epsilon / (dist + kernel_epsilon)
            s = np.sqrt(np.sum(kernel)) + c

            if np.isnan(s) or s > sm:
                counts[step] = 0.
            else:
                counts[step] = 1 / s
        return counts

    def compute_irs(self, buffer, time_steps):
        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        intrinsic_rewards = np.zeros_like(buffer.rewards)

        n_steps = buffer.observations.shape[0]
        n_envs = buffer.observations.shape[1]
        obs = torch.from_numpy(buffer.observations)
        obs = obs.to(self.device)
        with torch.no_grad():
            for idx in range(n_envs):
                if len(self.ob_shape) == 3:
                    encoded_obs = self.cnn_encoder(obs[:, idx, :, :, :])
                else:
                    encoded_obs = obs[:, idx]
                dist = torch.norm(encoded_obs[:-1] - encoded_obs[1:], p=2, dim=1)
                intrinsic_rewards[:-1, idx] = dist.cpu().numpy()

                n_eps = self.pseudo_count(encoded_obs)
                intrinsic_rewards[:-1, idx] = n_eps[1:] * intrinsic_rewards[:-1, idx]

        self.update(buffer)

        return beta_t * intrinsic_rewards

