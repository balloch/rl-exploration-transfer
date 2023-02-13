from typing import Union, List, Type, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F

from stable_baselines3.common.utils import get_device


from .noisy_layer import NoisyLinear


def create_noisy_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
    num_noisy_layers=2,
) -> List[nn.Module]:

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        if idx >= len(net_arch) - num_noisy_layers:
            modules.append(
                NoisyLinear(net_arch[idx], net_arch[idx + 1], bias=with_bias)
            )
        else:
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        if num_noisy_layers > 0:
            modules.append(NoisyLinear(last_layer_dim, output_dim, bias=with_bias))
        else:
            modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))

    if squash_output:
        modules.append(nn.Tanh())

    return modules


class NoisyMlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        # Iterate through the policy layers and build the policy net
        policy_net = create_noisy_mlp(
            input_dim=feature_dim,
            output_dim=pi_layers_dims[-1],
            net_arch=pi_layers_dims[:-1],
            activation_fn=activation_fn,
            num_noisy_layers=1,
        )

        # Iterate through the value layers and build the value net
        value_net = create_noisy_mlp(
            input_dim=feature_dim,
            output_dim=vf_layers_dims[-1],
            net_arch=vf_layers_dims[:-1],
            activation_fn=activation_fn,
            num_noisy_layers=0,
        )

        # Save dim, used to create the distributions
        self.latent_dim_pi = pi_layers_dims[-1]
        self.latent_dim_vf = vf_layers_dims[-1]

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
