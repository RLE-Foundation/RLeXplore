<div align=center>
<br>
<img src='./assets/logo.png' style="width: 70%">
<br>

## RLeXplore: Accelerating Research in Intrinsically-Motivated Reinforcement Learning
</div>

**RLeXplore** is a unified, highly-modularized and plug-and-play toolkit that currently provides high-quality and reliable implementations of eight representative intrinsic reward algorithms. It used to be challenging to compare intrinsic reward algorithms due to various confounding factors, including distinct implementations, optimization strategies, and evaluation methodologies. Therefore, RLeXplore is designed to provide unified and standardized procedures for constructing, computing, and optimizing intrinsic reward modules.

The workflow of RLeXplore is illustrated as follows:
<div align=center>
<img src='./assets/workflow.png' style="width: 100%">
</div>

# Table of Contents
- [Installation](#installation)
- [Module List](#module-list)
- [Tutorials](#tutorials)
- [Benchmark Results](#benchmark-results)
- [Cite Us](#cite-us)

# Installation
- with pip `recommended`

Open a terminal and install **rllte** with `pip`:
``` shell
conda create -n rllte python=3.8
pip install rllte-core 
```

- with git

Open a terminal and clone the repository from [GitHub](https://github.com/RLE-Foundation/rllte) with `git`:
``` sh
git clone https://github.com/RLE-Foundation/rllte.git
pip install -e .
```

Now you can invoke the intrinsic reward module by:
``` python
from rllte.xplore.reward import ICM, RIDE, ...
```

## Module List
| **Type** 	| **Modules** 	|
|---	|---	|
| Count-based 	| [PseudoCounts](https://arxiv.org/pdf/2002.06038), [RND](https://arxiv.org/pdf/1810.12894.pdf), [E3B](https://proceedings.neurips.cc/paper_files/paper/2022/file/f4f79698d48bdc1a6dec20583724182b-Paper-Conference.pdf) 	|
| Curiosity-driven 	| [ICM](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf), [Disagreement](https://arxiv.org/pdf/1906.04161.pdf), [RIDE](https://arxiv.org/pdf/2002.12292) 	|
| Memory-based 	| [NGU](https://arxiv.org/pdf/2002.06038) 	|
| Information theory-based 	| [RE3](http://proceedings.mlr.press/v139/seo21a/seo21a.pdf) 	|

## Tutorials
Click the following links to get the code notebook:

0. [Quick Start](./0%20quick_start.ipynb)
1. [RLeXplore with RLLTE](./1%20rlexplore_with_rllte.ipynb)
2. [RLeXplore with Stable-Baselines3](./2%20rlexplore_with_sb3.ipynb)
3. [RLeXplore with CleanRL](./3%20rlexplore_with_cleanrl.py)
4. [Exploring Mixed Intrinsic Rewards](./4%20mixed_intrinsic_rewards.ipynb)
4. [Custom Intrinsic Rewards](./5%20custom_intrinsic_reward.ipynb)

## Benchmark Results
- `RLLTE's PPO+RLeXplore` on *SuperMarioBros*:

<div align=center>
<img src='./assets/smb.png' style="width: 100%">
</div>

- `RLLTE's PPO+RLeXplore` on *MiniGrid*:

  + DoorKey-16×16
  <div align=center>
  <img src='./assets/mgd.png' style="width: 100%">
  </div>

  + KeyCorridorS8R5, KeyCorridorS9R6, KeyCorridorS10R7, MultiRoom-N7-S8, MultiRoom-N10-S10, MultiRoom-N12-S10,	Dynamic-Obstacles-16x16,	and LockedRoom
  <div align=center>
  <img src='./assets/mg_hard.png' style="width: 100%">
  </div>

- `RLLTE's PPO+RLeXplore` on five hard-exploration tasks of *ALE*:

| **Algorithm** | **Gravitar** | **MontezumaRevenge** | **PrivateEye** | **Seaquest** | **Venture** |
|---------------|:------------:|:--------------------:|:--------------:|:------------:|:-----------:|
| Extrinsic     |  **1060.19** |         42.83        |      88.37     |    942.37    |    391.73   |
| Disagreement  |    689.12    |         0.00         |      33.23     |    6577.03   |    468.43   |
| E3B           |    503.43    |         0.50         |      66.23     |  **8690.65** |     0.80    |
| ICM           |    194.71    |         31.14        |     -27.50     |    2626.13   |     0.54    |
| PseudoCounts  |    295.49    |         0.00         |   **1076.74**  |    668.96    |     1.03    |
| RE3           |    130.00    |         2.68         |     312.72     |    864.60    |     0.06    |
| RIDE          |    452.53    |         0.00         |      -1.40     |    1024.39   |    404.81   |
| RND           |    835.57    |      **160.22**      |      45.85     |    5989.06   |  **544.73** |

- `CleanRL's PPO+RLeXplore's RND` on *Montezuma's Revenge*:

<div align=center>
<img src='./assets/atari_curves.png' style="width: 70%">
</div>


- `RLLTE's SAC+RLeXplore` on *Ant-UMaze*:

<div align=center>
<img src='./assets/sac_ant.png' style="width: 70%">
</div>

## Cite Us
To cite this repository in publications:

``` bib
@article{yuan_roger2024rlexplore,
  title={RLeXplore: Accelerating Research in Intrinsically-Motivated Reinforcement Learning},
  author={Yuan, Mingqi and Castanyer, Roger Creus and Li, Bo and Jin, Xin and Berseth, Glen and Zeng, Wenjun},
  journal={arXiv preprint arXiv:2405.19548},
  year={2024}
}
```
