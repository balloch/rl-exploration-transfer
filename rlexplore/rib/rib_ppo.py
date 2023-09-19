# Atharv Sonwane <atharvs.twm@gmail.com>
from typing import Union, List, Type, Tuple, Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

threeD = True
test = True
class Relational(nn.Module):
    def __init__(self, input_shape, nheads=1, hidden_dim=None, output_dim=None, use_sde =False):
        super(Relational, self).__init__()
        self.input_shape = input_shape
        self.nheads = nheads
        self.features = input_shape[-1]
        # self.features = input_shape[]
        print("input shape:",input_shape)
        # if hidden_dim is None:
        #     self.hidden_dim = self.features
        # else:
        #     self.hidden_dim = hidden_dim
        if hidden_dim is None:
            self.hidden_dim = self.input_shape[0]
        else:
            self.hidden_dim = hidden_dim
        if output_dim is None:
            self.output_dim = self.features
        else:
            self.output_dim = output_dim

        self.q_projection = nn.Linear(self.features, self.hidden_dim)
        self.k_projection = nn.Linear(self.features, self.hidden_dim)
        self.v_projection = nn.Linear(self.features, self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, self.output_dim)
        # self.output_linear = nn.Linear(4, self.output_dim)

    def forward(self, x):
        x = self._apply_self_attention(x)
        print(x.shape)
        x = self.output_linear(x)
        return x

    def _apply_self_attention(self, x):
        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        print("Original shape of q:", q.shape)
        if threeD:
            q = q.view(*q.shape[:-1], self.nheads, -1).transpose(-2, -3)
            k = k.view(*k.shape[:-1], self.nheads, -1).transpose(-2, -3)
            v = v.view(*v.shape[:-1], self.nheads, -1).transpose(-2, -3)
        else:
            q = q.view(*q.shape[:-1], self.nheads, -1).transpose(1, 2)
            k = k.view(*k.shape[:-1], self.nheads, -1).transpose(1, 2)
            v = v.view(*v.shape[:-1], self.nheads, -1).transpose(1, 2)
        print("shape of q after reshaping:", q.shape)

        d = torch.tensor([self.features], dtype=x.dtype)
        w = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(d), dim=-1)
        scores = torch.matmul(w, v)

        scores = scores.transpose(-2, -3)
        scores = scores.view(*scores.shape[:-2], -1)

        return scores


class RelationalActorCritic(nn.Module):
    def __init__(
        self,
        observation_space, #for obs_shape
        action_space, # for a_dim,
        conv_dims: List[int],
        feature_dim,
        lin_dims,
        relational_hidden_dim=None,
        relational_output_dim=None,
        use_sde =False
    ):
        super(RelationalActorCritic, self).__init__()
        if not test:
            self.obs_shape = observation_space.shape
            self.a_dim = action_space.n  # Assuming discrete action space
            print(f"observation shape before assignment: {self.obs_shape}")
            print(f"observation shape before assignment: {self.a_dim}")
            print(f"conv_dims type before assignment: {type(conv_dims)}")
            self.conv_dims = [8]
            conv_dims = self.conv_dims
            print(f"self.conv_dims type after assignment: {type(self.conv_dims)}")
            self.feature_dim = self.obs_shape[-1]  # Extracting the feature dimension
            self.lin_dims = [self.feature_dim, self.a_dim]  # A simple way to define lin_dims
        else:
            self.obs_shape = observation_space  # env.observation_space.shape
            self.a_dim = action_space  # env.action_space.n
            self.conv_dims = conv_dims
            self.feature_dim = feature_dim
            self.lin_dims = lin_dims
        
        self.conv_dims.insert(0, self.obs_shape[0])
        self.conv_dims.append(feature_dim)
        conv_module_list = []
        for i in range(len(self.conv_dims) - 1):
            conv_module_list.append(nn.Conv2d(conv_dims[i], conv_dims[i + 1], 2, 1))
            conv_module_list.append(nn.ReLU())
            conv_module_list.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*conv_module_list)

        if threeD:
            var = torch.zeros(4, *self.obs_shape, requires_grad=False)
            var = self.conv(var)
            var = var.flatten(start_dim=-2).transpose(-1, -2)
            c = torch.zeros(*var.shape[:-1], 1)
            var = torch.cat([var, c, c], dim=-1)
            
            self.relational = Relational(
                tuple(var.shape[-2:]),
                hidden_dim=relational_hidden_dim,
                output_dim=relational_output_dim,
            )
        else:
            # print(self.obs_shape[0])
            # print(*self.obs_shape)
            self.fc1 = nn.Linear(self.obs_shape[0], 32)
            var = torch.zeros(4, *self.obs_shape, requires_grad=False)
            var = self.fc1(var)

            # Assuming you want to continue concatenating position and velocity as separate features:
            c = torch.zeros(*var.shape[:-1], 1)
            var = torch.cat([var, c, c], dim=-1)
            
            self.relational = Relational(
                tuple(var.shape[-2:]),
                hidden_dim=relational_hidden_dim,
                output_dim=relational_output_dim,
            )
        
        

        print(var.shape)
        var = self.relational(var)
        var = torch.max(var, dim=-2).values
        lin_dims.insert(0, var.shape[-1])
        lin_dims.append(self.a_dim)
        lin_module_list = []
        for i in range(len(lin_dims) - 1):
            lin_module_list.append(nn.Linear(self.lin_dims[i], self.lin_dims[i + 1]))
            lin_module_list.append(nn.ReLU())
        self.linear = nn.Sequential(*lin_module_list)
        self.policy_head = nn.Linear(self.a_dim, self.a_dim)
        self.baseline_head = nn.Linear(self.a_dim, 1)

    def forward(self, x):
        if threeD:
            x = self.conv(x)
            ncols = x.shape[-1]
            x = x.flatten(start_dim=-2).transpose(-1, -2)
            c = torch.arange(x.shape[-2]).expand(*x.shape[:-1]).to(x.dtype)
            x_coord = (c % ncols).view(*x.shape[:-2], -1, 1)
            y_coord = (c // ncols).view(*x.shape[:-2], -1, 1)
            x = torch.cat([x, x_coord, y_coord], dim=-1)
            print("x shape:", x.shape)
            x = self.relational(x)
            x = torch.max(x, dim=-2).values
            x = self.linear(x)
            b = self.baseline_head(x)
            pi_logits = self.policy_head(x)
            return pi_logits, b
        else:
            x_coord = torch.tensor([0.0]).expand(*x.shape[:-1], -1).to(x.dtype)
            y_coord = torch.tensor([1.0]).expand(*x.shape[:-1], -1).to(x.dtype)
            x = torch.cat([x, x_coord, y_coord], dim=-1)
            print("x shape:", x.shape)
            print(x)
            #today's end point
            x = self.relational(x)
    
            # No need for the torch.max operation in this context
            x = self.linear(x)
    
            b = self.baseline_head(x)
            pi_logits = self.policy_head(x)
    
            return pi_logits, b

    def evaluate(self, obs, deterministic=False):
        pi_logits, b = self.forward(obs)
        if deterministic:
            a = torch.distributions.Categorical(logits=pi_logits).sample()
        else:
            a = torch.argmax(pi_logits)
        return a.item(), pi_logits, b

    def sample_action(self, obs, deterministic=False):
        a, _, _ = self.evaluate(obs, deterministic)
        return a


if __name__ == "__main__":
    ac = RelationalActorCritic((3, 210, 160), 2, [8], 8, [8])
    x = torch.rand(3,210,160)

    # ac = RelationalActorCritic((2,), 3, [8], 8, [8])
    # x = torch.randn(2,)
    pi_logits, b = ac(x)
    a = ac.sample_action(x)
    print(a, pi_logits.shape, b.shape)