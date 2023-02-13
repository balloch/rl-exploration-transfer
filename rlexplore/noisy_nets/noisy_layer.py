import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        std_init: float = 0.5,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.bias = bias

        self.weight_mu = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(self.out_features, self.in_features)
        )

        self.register_buffer(
            "weight_epsilon", torch.Tensor(self.out_features, self.in_features)
        )

        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(self.out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(self.out_features))
            self.register_buffer("bias_epsilon", torch.Tensor(self.out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))

        if self.bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = NoisyLinear.scale_noise(self.in_features)
        epsilon_out = NoisyLinear.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        if self.bias:
            self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        if train:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon if self.bias else 0,
            )
        else:
            return F.linear(
                x,
                self.weight_mu,
                self.bias_mu if self.bias else 0,
            )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())
