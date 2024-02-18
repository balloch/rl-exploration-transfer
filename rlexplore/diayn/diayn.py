from rlexplore.networks.random_encoder import MlpEncoder, CnnEncoder


from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import gymnasium as gym


class Diayn(object):

    def __init__(self, 
                 envs, 
                 device, 
                 lr,
                 batch_size,
                 beta,
                 kappa,
                 skill_key="skill",
                 state_key="state",
                 ) -> None:
        """
        Diversity Is All You Need: Learning Skills Without a Reward Function (Adapted with task reward)
        Paper: https://arxiv.org/pdf/1802.06070.pdf

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param lr: The learning rate of the discriminator model.
        :param batch_size: The batch size to train the discriminator model.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """
        self.device = device
        self.beta = beta
        self.kappa = kappa
        self.lr = lr
        self.batch_size = batch_size
        self.skill_key = skill_key
        self.state_key = state_key

        assert isinstance(envs.observation_space, gym.spaces.Dict)
        assert self.skill_key in envs.observation_space.keys()
        assert self.state_key in envs.observation_space.keys()

        self.ob_shape = envs.observation_space[self.state_key].shape
        if isinstance(envs.observation_space[self.skill_key], gym.spaces.Discrete):
            self.skill_shape = envs.observation_space[self.skill_key].n
            self.skill_type = gym.spaces.Discrete
            self.discriminator_loss = nn.CrossEntropyLoss()
        elif isinstance(envs.observation_space[self.skill_key], gym.spaces.Box):
            self.skill_shape = envs.observation_space[self.skill_key].shape
            self.skill_type = gym.spaces.Box
            assert len(self.skill_shape) == 1
            self.skill_shape = self.skill_shape[0]
            self.discriminator_loss = nn.MSELoss()
        else:
            raise NotImplementedError


        if len(self.ob_shape) == 3:
            self.discriminator = CnnEncoder(self.ob_shape, self.skill_shape).to(self.device)
        else:
            self.discriminator = MlpEncoder(self.ob_shape, self.skill_shape).to(self.device)

        if self.skill_type == gym.spaces.Discrete:
            self.discriminator.main.append(nn.Softmax(-1))

        self.optimizer = optim.Adam(lr=self.lr, params=self.discriminator.parameters())

    def update(self, rollouts):
        n_steps = rollouts["observations"][self.state_key].shape[0]
        n_envs = rollouts["observations"][self.state_key].shape[1]

        obs = torch.from_numpy(rollouts["observations"][self.state_key]).reshape(n_steps * n_envs, *self.ob_shape)
        if self.skill_type == gym.spaces.Discrete:
            skills = torch.from_numpy(rollouts["observations"][self.skill_key]).reshape((n_steps * n_envs, ))
            skills = F.one_hot(skills.to(torch.int64), self.skill_shape).float()
        elif self.skill_type == gym.spaces.Box:
            skills = torch.from_numpy(rollouts["observations"][self.skill_key]).reshape(n_steps * n_envs, self.skill_shape)
        
        obs = obs.to(self.device)
        skills = skills.to(self.device)

        dataset = TensorDataset(obs, skills)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for (batch_obs, batch_skills) in loader:
            pred_skills = self.discriminator(batch_obs)

            loss = self.discriminator_loss(pred_skills, batch_skills)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_irs(self, rollouts, time_steps):
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        n_steps = rollouts["observations"][self.state_key].shape[0]
        n_envs = rollouts["observations"][self.state_key].shape[1]

        intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))

        obs = torch.from_numpy(rollouts["observations"][self.state_key])
        skills = torch.from_numpy(rollouts["observations"][self.skill_key])

        if self.skill_type == gym.spaces.Discrete:
            skills = F.one_hot(skills[:, :, 0].to(torch.int64), self.skill_shape).float()
        
        obs = obs.to(self.device)
        skills = skills.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                discriminator_output = self.discriminator(obs[:, idx])
                true_skill = skills[:, idx]
                if self.skill_type == gym.spaces.Discrete:
                    intrinsic_rewards[:, idx] = np.log(torch.sum(true_skill * discriminator_output, axis=1).cpu().numpy())[:, np.newaxis] - np.log(1 / self.skill_shape)
                elif self.skill_type == gym.spaces.Box:
                    intrinsic_rewards[:, idx] = -F.mse_loss(discriminator_output, true_skill, reduction="mean").cpu().numpy()

        self.update(rollouts)

        return beta_t * intrinsic_rewards