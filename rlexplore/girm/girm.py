#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：girm.py
@Author ：YUAN Mingqi
@Date ：2022/9/21 14:16 
'''

from rlexplore.networks.girm_vae_encoder_decoder import CnnEncoder, CnnDecoder
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self,
                 device,
                 model_base,
                 in_channels,
                 latent_dim,
                 action_dim,
                 ):
        super(VAE, self).__init__()
        if model_base == 'cnn':
            self.encoder = CnnEncoder(
                kwargs={'in_channels': in_channels * 2, 'latent_dim': latent_dim, 'action_dim': action_dim}
            )
            self.decoder = CnnDecoder(
                kwargs={'latent_dim': latent_dim, 'action_dim': action_dim, 'out_channels': in_channels}
            )

        self.mu = nn.Linear(latent_dim, action_dim)
        self.logvar = nn.Linear(latent_dim, action_dim)

        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def reparameterize(self, mu, logvar, device, training=True):
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_()).to(device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, obs, next_obs):
        latent = self.encoder(obs, next_obs)
        mu = self.mu(latent)
        logvar = self.logvar(latent)

        z = self.reparameterize(mu, logvar, self.device)

        reconstructed_next_obs = self.decoder(z, obs)

        return z, mu, logvar, reconstructed_next_obs

class GIRM(object):
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
        Intrinsic Reward Driven Imitation Learning via Generative Model (GIRM)
        Paper: http://proceedings.mlr.press/v119/yu20d/yu20d.pdf

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
            self.action_dim = self.action_shape
        elif envs.action_space.__class__.__name__ == 'Box':
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.shape
            self.action_dim = self.action_shape[0]
        else:
            raise NotImplementedError
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.kappa = kappa

        if len(self.ob_shape) == 3:
            model_base = 'cnn'
            self.vae = VAE(
                device=device,
                model_base=model_base,
                action_dim=self.action_dim,
                in_channels=self.ob_shape[0],
                latent_dim=latent_dim
            )
        self.vae.to(self.device)
        self.optimizer = optim.Adam(lr=lr, params=self.vae.parameters())

    def compute_irs(self, buffer, time_steps):
        