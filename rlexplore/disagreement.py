# =============================================================================
# MIT License

# Copyright (c) 2024 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================



from typing import Dict, Optional
import numpy as np
import gymnasium as gym
import torch as th
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from rllte.common.prototype import BaseReward
from .model import ObservationEncoder, ForwardDynamicsModel


class Disagreement(BaseReward):
    """Self-Supervised Exploration via Disagreement (Disagreement).
        See paper: https://arxiv.org/pdf/1906.04161.pdf

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        rwd_norm_type (bool): Use running mean and std for reward normalization.
        obs_rms (bool): Use running mean and std for observation normalization.
        gamma (Optional[float]): Intrinsic reward discount rate, None for no discount.
        latent_dim (int): The dimension of encoding vectors.
        n_envs (int): The number of parallel environments.
        lr (float): The learning rate.
        batch_size (int): The batch size for training.
        ensemble_size (int): The number of forward dynamics models in the ensemble.
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.

    Returns:
        Instance of Disagreement.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        latent_dim: int = 128,
        lr: float = 0.001,
        rwd_norm_type: str = "rms",
        obs_rms: bool = True,
        gamma: Optional[float] = None,
        n_envs: int = 1,
        batch_size: int = 256,
        ensemble_size: int = 4,
        update_proportion: float = 1.0,
        encoder_model: str = "mnih",
        weight_init: str = "default"
    ) -> None:
        super().__init__(observation_space, action_space, n_envs, device, beta, kappa, rwd_norm_type, obs_rms, gamma)
        
        self.random_encoder = ObservationEncoder(obs_shape=self.obs_shape,
                                                   latent_dim=latent_dim, encoder_model=encoder_model, weight_init=weight_init).to(self.device)

        # freeze the randomly initialized target network parameters
        for p in self.random_encoder.parameters():
            p.requires_grad = False

        # build the ensemble of forward dynamics models
        self.ensemble_size = ensemble_size
        self.ensemble = [
            ForwardDynamicsModel(latent_dim=latent_dim,
                                    action_dim=self.policy_action_dim, encoder_model=encoder_model).to(self.device)
            for _ in range(self.ensemble_size)
        ]        
        self.opt = [
            th.optim.Adam(self.ensemble[i].parameters(), lr=lr)
            for i in range(self.ensemble_size)
        ]
        # set the parameters
        self.batch_size = batch_size
        self.update_proportion = update_proportion

    def watch(self, 
              observations: th.Tensor, 
              actions: th.Tensor,
              rewards: th.Tensor,
              terminateds: th.Tensor,
              truncateds: th.Tensor,
              next_observations: th.Tensor
              ) -> Optional[Dict[str, th.Tensor]]:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): Observations data with shape (n_envs, *obs_shape).
            actions (th.Tensor): Actions data with shape (n_envs, *action_shape).
            rewards (th.Tensor): Extrinsic rewards data with shape (n_envs).
            terminateds (th.Tensor): Termination signals with shape (n_envs).
            truncateds (th.Tensor): Truncation signals with shape (n_envs).
            next_observations (th.Tensor): Next observations data with shape (n_envs, *obs_shape).

        Returns:
            Feedbacks for the current samples, e.g., intrinsic rewards for the current samples. This 
            is useful when applying the memory-based methods to off-policy algorithms.
        """
        
    def compute(self, samples: Dict[str, th.Tensor]) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors, whose keys are
            'observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'. For example, 
            the data shape of 'observations' is (n_steps, n_envs, *obs_shape). 

        Returns:
            The intrinsic rewards.
        """
        super().compute(samples)
        
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("observations").size()[:2]
        # get the observations
        obs_tensor = samples.get("observations").to(self.device).view(-1, *self.obs_shape)
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        actions_tensor = samples.get("actions").to(self.device).view(-1, *self.action_shape)
        # apply one-hot encoding if the action type is discrete
        if self.action_type == "Discrete":
            actions_tensor = F.one_hot(actions_tensor.long(), self.policy_action_dim).float()
        # compute the intrinsic rewards
        with th.no_grad():
            random_feats = self.random_encoder(obs_tensor.view(-1, *self.obs_shape))
            preds = []
            for i in range(self.ensemble_size):
                next_obs_hat = self.ensemble[i](random_feats, actions_tensor)
                preds.append(next_obs_hat)
            preds = th.stack(preds, dim=0)
            intrinsic_rewards = th.var(preds, dim=0).mean(dim=-1).view(n_steps, n_envs)
        
        # update the reward module
        self.update(samples)
            
        # return the scaled intrinsic rewards
        return self.scale(intrinsic_rewards)


    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.
                The `update` function will be invoked after the `compute` function.

        Returns:
            None.
        """
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        # get the observations and actions
        obs_tensor = samples.get("observations").to(self.device).view(-1, *self.obs_shape)
        next_obs_tensor = samples.get("next_observations").to(self.device).view(-1, *self.obs_shape)
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        next_obs_tensor = self.normalize(next_obs_tensor)
        # apply one-hot encoding if the action type is discrete
        if self.action_type == "Discrete":
            actions_tensor = samples.get("actions").view(n_steps * n_envs)
            actions_tensor = F.one_hot(actions_tensor.long(), self.policy_action_dim).float()
        else:
            actions_tensor = samples.get("actions").view(n_steps * n_envs, -1)
        # build the dataset and dataloader
        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        
        avg_loss = []
        # update the ensemble of forward dynamics models
        for _idx, batch_data in enumerate(loader):
            ensemble_idx = _idx % self.ensemble_size
            # get the batch data
            obs, actions, next_obs = batch_data
            obs, actions, next_obs = obs.to(self.device), actions.to(self.device), next_obs.to(self.device)
            # zero the gradients
            self.opt[ensemble_idx].zero_grad()
            # get the encoded observations and next observations
            with th.no_grad():
                encoded_obs = self.random_encoder(obs)
                encoded_next_obs = self.random_encoder(next_obs)
            # compute the predicted next observations
            pred_next_obs = self.ensemble[ensemble_idx](encoded_obs, actions)
            # compute the forward dynamics loss
            fm_loss = F.mse_loss(pred_next_obs, encoded_next_obs, reduction="none").mean(-1)
            # use a random mask to select a subset of the training data
            mask = th.rand(len(fm_loss), device=self.device)
            mask = (mask < self.update_proportion).type(th.FloatTensor).to(self.device)
            # get the masked loss
            fm_loss = (fm_loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            # backward and update
            fm_loss.backward()
            self.opt[ensemble_idx].step()
            
            avg_loss.append(fm_loss.item())
            
        self.logger.record("avg_loss", np.mean(avg_loss))