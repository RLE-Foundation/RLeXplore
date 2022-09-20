#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：envs.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 20:04 
'''

import gym
import numpy as np

from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

try:
    import pybullet_envs
except:
    pass

try:
    import dmc2gym
except:
    pass

class DMCEnv(gym.Env):
    def __init__(self, env_id):
        _, domain, task = env_id.split('.')
        self.env = dmc2gym.make(domain_name=domain, task_name=task)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high).astype('float32')
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

def create_env(env_id, n_envs, log_dir):
    if 'dm' in env_id:
        envs = make_vec_env(
            env_id=DMCEnv,
            n_envs=10,
            monitor_dir=log_dir,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={'env_id': env_id}
        )
    elif 'Bullet' in env_id:
        envs = make_vec_env(
            env_id=env_id,
            n_envs=n_envs,
            monitor_dir=log_dir,
            vec_env_cls=SubprocVecEnv,
        )
    else:
        # Atari games
        envs = make_atari_env(
            env_id=env_id,
            n_envs=n_envs,
            monitor_dir=log_dir,
            vec_env_cls=SubprocVecEnv,
        )
        envs = ClipRewardEnv(envs)
        envs = VecFrameStack(envs, n_stack=4)
        envs = VecTransposeImage(envs)

    return envs