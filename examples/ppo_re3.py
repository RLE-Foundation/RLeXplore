#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：ppo_re3.py
@Author ：YUAN Mingqi
@Date ：2022/9/19 21:29 
'''

import torch
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from rlexplore.re3 import RE3

if __name__ == '__main__':
    device = torch.device('cuda:0')
    env_id = 'AntBulletEnv-v0'
    n_envs = 2
    n_steps = 128
    # Create vectorized environments
    envs = make_vec_env(
            env_id=env_id,
            n_envs=n_envs,
            monitor_dir='./logs',
            vec_env_cls=SubprocVecEnv,
        )
    # Create RE3 module
    re3 = RE3(envs=envs, device=device, embedding_size=64, beta=1e-2, kappa=1e-5)
    # Compute intrinsic rewards for random observations and random time steps
    intrinsic_rewards = re3.compute_irs(
        obs_array=np.random.rand(n_envs, n_steps, envs.observation_space.shape[0]), # ((n_steps, n_envs) + obs_shape)
        time_steps=10000,
        k=3
    )