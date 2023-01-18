<div align=center>
<img src='./docs/logo.jpg'>
</div>

<img src="https://img.shields.io/badge/Python->=3.8-brightgreen"> <img src="https://img.shields.io/badge/PyTorch->=1.8.1-orange"> <img src="https://img.shields.io/badge/Gym->=0.21.1-%23252422"> <img src="https://img.shields.io/badge/PyBullet-3.2.5-%2306d6a0">  <img src="https://img.shields.io/badge/DMC Suite-1.0.5-blue"> <img src="https://img.shields.io/badge/JAX-0.3.17-%238338ec"> <img src="https://img.shields.io/badge/Docs-Developing-%23ff595e"> 


# Reinforcement Learning Exploration Baselines (RLeXplore)

RLeXplore is a set of implementations of intrinsic reward driven-exploration approaches in reinforcement learning using PyTorch, which can be deployed in arbitrary algorithms in a plug-and-play manner. In particular, RLeXplore is
designed to be well compatible with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), providing more stable exploration benchmarks. 

<div align=center>
<img src='./docs/flowchart.png' style="width: 600px">
</div>

- See [Changelog](#changelog) and [Implemented Algorithms](#implemented-algorithms);
- Code test in progress! Welcome to contribute to this program!

# Installation
- Get the repository with git:
```
git clone https://github.com/yuanmingqi/rl-exploration-baselines.git
```
- Run the following command to get dependencies:
```shell
pip install -r requirements.txt
```

# Usage Example
Due to the large differences in the calculation of different intrinsic reward methods, RLeXplore has the following rules:

1. In RLeXplore, the environments are assumed to be ***vectorized***;
2. The ***compute_irs*** function of each intrinsic reward module has a mandatory argument ***rollouts***, which is a dict like:
```diff
+ observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>
- actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>
+ rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>
```

Take RE3 for instance, it computes the intrinsic reward for each state based on the Euclidean distance between the state and 
its $k$-nearest neighbor within a mini-batch. Thus it suffices to provide ***observations</font>*** data to compute the reward. The following code provides a usage example of RE3:
```python
import torch
import numpy as np
from rlexplore.re3 import RE3

if __name__ == '__main__':
    ''' env setup '''
    device = torch.device('cuda:0')
    obs_shape = (4, 84, 84)
    action_shape = 1 # for discrete action space
    n_envs = 16 
    n_steps = 256 
    observations = np.random.randn(
       n_steps, n_envs, *obs_shape).astype('float32') # collected experiences 

    ''' create RE3 instance '''
    re3 = RE3(obs_shape=obs_shape, action_shape=action_shape, device=device,
              latent_dim=128, beta=0.05, kappa=0.00001)

    ''' compute intrinsic rewards '''
    intrinsic_rewards = re3.compute_irs(rollouts={'observations': observations},
        time_steps=25600, k=3, average_entropy=False)

    print(intrinsic_rewards.shape, type(intrinsic_rewards))
    print(intrinsic_rewards)

# Output: (256, 16, 1) <class 'numpy.ndarray'>
```

Train with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) on PyBullet games:
```shell
python examples/ppo_re3_bullet.py --action-space cont --env-id AntBulletEnv-v0 --algo ppo --n-envs 10 --exploration re3 --total-time-steps 2000000 --n-steps 128
```

# Implemented Algorithms
| Algorithm | Remark                                  | Year  | Paper                                                                                                                                                                                                                                                                      | Code                                                                                    |
|:----------|:----------------------------------------|:------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|
| ICM       | Curiosity-driven exploration                 | 2017  | [Curiosity-Driven Exploration by Self-Supervised Prediction](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf)                                                                                                                                                     | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/icm)  |
| RND       | Count-based exploration            | 2019  | [Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)                                                                                                                                                                                         | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/rnd)  |
| GIRM      | Curiosity-driven exploration            | 2020  | [Intrinsic Reward Driven Imitation Learning via Generative Model](http://proceedings.mlr.press/v119/yu20d/yu20d.pdf)                                                                                                                                                       | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/girm) |
| NGU       | Memory-based exploration                | 2020  | [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/pdf/2002.06038)                                                                                                                                                                                | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/ngu)  |
| RIDE      | Procedurally-generated environment      | 2020  | [RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments](https://arxiv.org/pdf/2002.12292)                                                                                                                                                      | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/ride) |
| RE3       | Shannon Entropy Maximization            | 2021  | [State Entropy Maximization with Random Encoders for Efficient Exploration](http://proceedings.mlr.press/v139/seo21a/seo21a.pdf)                                                                                                                                           | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/re3)  |
| RISE      | Rényi Entropy Maximization              | 2022  | [Rényi State Entropy Maximization for Exploration Acceleration in Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9802917/)                                                                                                                          | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/rise) |
| REVD      | Rényi Divergence Maximization           | 2022  | [Rewarding Episodic Visitation Discrepancy for Exploration in Reinforcement Learning](https://openreview.net/pdf?id=V2pw1VYMrDo)                                                                                                                                           | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/revd) |

# Changelog
**28/12/2022**
- Update RE3, RISE, RND, RIDE.
- Add a new method entitled REVD.

**04/12/2022**
- Update RND and RIDE.

**03/12/2022**
- We start to reconstruct the project to make it compatible with arbitrary tasks;
- Update RE3 and RISE.

**27/09/2022**
- Update the RISE;
- Introduce JAX in RISE. See ```experimental``` folder.

**26/09/2022**
- Update the RE3;
- Try to introduce JAX to accelerate computation. See ```experimental``` folder.

# Acknowledgments
Some source codes of RLeXplore are built based on the following repositories:

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [GIRIL](https://github.com/xingruiyu/GIRIL)
- [never-give-up](https://github.com/Coac/never-give-up)
