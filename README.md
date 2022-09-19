
<div style="align: center;">
<img src='./docs/logo.jpg'>
</div>

<img src="https://img.shields.io/badge/Building-Pass-brightgreen"> <img src="https://img.shields.io/badge/Framework-PyTorch-orange"> <img src="https://img.shields.io/badge/Docs-Developing-blue">


# Reinforcement Learning Exploration Baselines (RLeXplore)

RLeXplore is a set of implementations of exploration approaches in reinforcement learning using PyTorch, which can be deployed in arbitrary algorithms in a plug-and-play manner. 

# Implemented Algorithms
| Algorithm | Remark | Paper | Code     |
| :-----| :----- | :-----  |:---------|
|GIRM|IRS|[Intrinsic Reward Driven Imitation Learning via Generative Model](http://proceedings.mlr.press/v119/yu20d/yu20d.pdf)| [Link]() |
|ICM|IRS|[Curiosity-Driven Exploration by Self-Supervised Prediction](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf)| [Link]() |
| RE3 | IRS | [State Entropy Maximization with Random Encoders for Efficient Exploration](http://proceedings.mlr.press/v139/seo21a/seo21a.pdf) | [Link]() |
|RIDE|IRS|[RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments](https://arxiv.org/pdf/2002.12292)| [Link]() |
|RND|IRS|[Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf%20http://arxiv.org/abs/1810.12894)| [Link]() |


# Installation
- Get the repository with git:
```
git clone https://github.com/yuanmingqi/rl-exploration-baselines.git
```
Run the following command to get dependencies:
```shell
pip install -r requirements.txt
```

# Usage Example
```python
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
```

# Acknowledgments
Some source codes of RLeXplore are built based on the following repositories:

- [re3]()
- []()
