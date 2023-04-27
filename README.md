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

# Notice
**This repo has been merged with a new project: [https://github.com/RLE-Foundation/Hsuanwu](https://github.com/RLE-Foundation/Hsuanwu), in which more reasonable implementations are provided!**

Invoke the intrinsic reward module by:
``` python
from hsuanwu.xplore.reward import ICM, RIDE, ...
```

## Module List
| Module | Remark | Repr.  | Visual | Reference | 
|:-|:-|:-|:-|:-|
| PseudoCounts | Count-Based exploration |✔️|✔️|[Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/pdf/2002.06038) |
| ICM  | Curiosity-driven exploration  | ✔️|✔️| [Curiosity-Driven Exploration by Self-Supervised Prediction](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf) | 
| RND  | Count-based exploration  | ❌|✔️| [Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf) | 
| GIRM | Curiosity-driven exploration  | ✔️ |✔️| [Intrinsic Reward Driven Imitation Learning via Generative Model](http://proceedings.mlr.press/v119/yu20d/yu20d.pdf)|
| NGU | Memory-based exploration  | ✔️  |✔️| [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/pdf/2002.06038) | 
| RIDE| Procedurally-generated environment | ✔️ |✔️| [RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments](https://arxiv.org/pdf/2002.12292)|
| RE3  | Entropy Maximization | ❌ |✔️| [State Entropy Maximization with Random Encoders for Efficient Exploration](http://proceedings.mlr.press/v139/seo21a/seo21a.pdf) |
| RISE  | Entropy Maximization  | ❌  |✔️| [Rényi State Entropy Maximization for Exploration Acceleration in Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9802917/) | 
| REVD  | Divergence Maximization | ❌  |✔️| [Rewarding Episodic Visitation Discrepancy for Exploration in Reinforcement Learning](https://openreview.net/pdf?id=V2pw1VYMrDo)|
|ProtoRL<sup>🐌</sup>| Entropy Maximization | ✔️ | ✔️ | [Reinforcement Learning with Prototypical Representations](http://proceedings.mlr.press/v139/yarats21a/yarats21a.pdf) |
|APS<sup>🐌</sup>| Skill Discovery | ✔️ | ✔️ | [APS: Active Pretraining with Successor Features](http://proceedings.mlr.press/v139/liu21b/liu21b.pdf) |

> - 🐌: Developing.
> - `Repr.`: The method involves representation learning.
> - `Visual`: The method works well in visual RL.
