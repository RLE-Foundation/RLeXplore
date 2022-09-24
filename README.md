
<div style="text-align: center;">
<img src='./docs/logo.jpg'>
</div>

<img src="https://img.shields.io/badge/Python->=3.8-brightgreen"> <img src="https://img.shields.io/badge/PyTorch->=1.8.1-orange"> <img src="https://img.shields.io/badge/Gym->=0.21.1-%23252422"> <img src="https://img.shields.io/badge/Pybullet-3.2.5-%2306d6a0">  <img src="https://img.shields.io/badge/DMC Suite-1.0.5-blue"> <img src="https://img.shields.io/badge/Docs-Developing-%23ff595e"> 


# Reinforcement Learning Exploration Baselines (RLeXplore)

RLeXplore is a set of implementations of exploration approaches in reinforcement learning using PyTorch, which can be deployed in arbitrary algorithms in a plug-and-play manner. In particular, RLeXplore is
designed to be well compatible with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), providing more stable exploration benchmarks.

Code test in progress! Welcome to contribute to this program!

# Implemented Algorithms
| Algorithm | Remark                             | Year | Paper                                                                                                                                             | Code                                                                                    |
|:----------|:-----------------------------------|:-----|:--------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|
| ICM       | Prediction-based exploration       | 2017 | [Curiosity-Driven Exploration by Self-Supervised Prediction](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf)                            | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/icm)  |
| RND       | Novelty-based exploration          | 2019 | [Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)                                                                | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/rnd)  |
| GIRM      | Prediction-based exploration       | 2020 | [Intrinsic Reward Driven Imitation Learning via Generative Model](http://proceedings.mlr.press/v119/yu20d/yu20d.pdf)                              | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/girm) |
| NGU       | Memory-based exploration           | 2020 | [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/pdf/2002.06038)                                                       | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/ngu)  |
| RIDE      | Procedurally-generated environment | 2020 | [RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments](https://arxiv.org/pdf/2002.12292)                             | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/ride) |
| RE3       | Computation-efficient exploration  | 2021 | [State Entropy Maximization with Random Encoders for Efficient Exploration](http://proceedings.mlr.press/v139/seo21a/seo21a.pdf)                  | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/re3)  |
| RISE      | Computation-efficient exploration  | 2022 | [Rényi State Entropy Maximization for Exploration Acceleration in Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9802917/) | [Link](https://github.com/yuanmingqi/rl-exploration-baselines/tree/main/rlexplore/rise) |

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
The following code illustrates how to use RLeXplore with Stable-Baselines3:
```python
import torch
from stable_baselines3 import PPO
from rlexplore.re3 import RE3
from rlexplore.utils import create_env

if __name__ == '__main__':
    device = torch.device('cuda:0')
    env_id = 'AntBulletEnv-v0'
    n_envs = 2
    n_steps = 128
    total_time_steps = 10000
    num_episodes = int(total_time_steps / n_steps / n_envs)
    # Create vectorized environments.
    envs = create_env(
            env_id=env_id,
            n_envs=n_envs,
            log_dir='./logs'
        )
    # Create RE3 module.
    re3 = RE3(envs=envs, device=device, latent_dim=64, beta=1e-2, kappa=1e-5)
    # Create PPO agent.
    model = PPO(policy='MlpPolicy', env=envs, n_steps=n_steps)
    _, callback = model._setup_learn(total_timesteps=total_time_steps, eval_env=None)

    for i in range(num_episodes):
        model.collect_rollouts(
            env=envs,
            rollout_buffer=model.rollout_buffer,
            n_rollout_steps=n_steps,
            callback=callback
        )
        # Compute intrinsic rewards.
        intrinsic_rewards = re3.compute_irs(
            buffer=model.rollout_buffer,
            time_steps=i * n_steps * n_envs,
            k=3)
        model.rollout_buffer.rewards += intrinsic_rewards
        # Update policy using the currently gathered rollout buffer.
        model.train()
```

# Acknowledgments
Some source codes of RLeXplore are built based on the following repositories:

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [GIRIL](https://github.com/xingruiyu/GIRIL)
- [never-give-up](https://github.com/Coac/never-give-up)
