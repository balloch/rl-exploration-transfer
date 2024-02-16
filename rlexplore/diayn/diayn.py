from rlexplore.networks.random_encoder import MlpEncoder, CnnEncoder


from torch import nn, optim
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
        elif isinstance(envs.observation_space[self.skill_key], gym.spaces.Box):
            self.skill_shape = envs.observation_space[self.skill_key].shape
            assert len(self.skill_shape) == 1
            self.skill_shape = self.skill_shape[0]
        else:
            raise NotImplementedError

        self.dm_loss = nn.MSELoss()

        if len(self.ob_shape) == 3:
            self.discriminator = CnnEncoder(self.ob_shape, self.skill_shape).to(self.device)
        else:
            self.discriminator = MlpEncoder(self.ob_shape, self.skill_shape).to(self.device)

        self.optimizer = optim.Adam(lr=self.lr, params=self.discriminator.parameters())

    def compute_irs(self, rollouts, time_steps):
        import pdb; pdb.set_trace()
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        n_steps = rollouts['observations'].shape[0]
        n_envs = rollouts['observations'].shape[1]

        intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))


        return beta_t * intrinsic_rewards