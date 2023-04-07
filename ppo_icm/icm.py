import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Set seed
torch.manual_seed(182)

class ICM(nn.Module):
    def __init__(self, state_size, action_size, icm_parameters):
        super(ICM, self).__init__()

        feature_hidden_sizes, feature_size, inverse_hidden_sizes, forward_hidden_sizes, self.β = icm_parameters
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Feature NN
        self.feature_net = nn.Sequential(
            nn.Linear(state_size, feature_hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(feature_hidden_sizes[0], feature_hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(feature_hidden_sizes[1], feature_size)
        )
        
        # Inverse NN
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_size * 2, inverse_hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(inverse_hidden_sizes[0], inverse_hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(inverse_hidden_sizes[1], action_size)
        )
        
        # Forward NN
        self.forward_net = nn.Sequential(
            nn.Linear(feature_size + action_size, forward_hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(forward_hidden_sizes[0], forward_hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(forward_hidden_sizes[1], feature_size)
        )

        self.L_I = nn.MSELoss()
        self.L_F = nn.MSELoss()
        
    def predict(self, s, a, s_prime):
        '''Predicts next action and state.'''
        s_reshaped = s.view(-1,self.state_size)
        a_reshaped = a.view(-1,self.action_size)
        s_prime_reshaped = s_prime.view(-1,self.state_size)
        cat1 = torch.cat([self.feature_net(s_reshaped), self.feature_net(s_prime_reshaped)], 1)
        cat2 = torch.cat([self.feature_net(s_reshaped).clone().detach(), a_reshaped], 1)
        a_hat = self.inverse_net(cat1)
        s_hat = self.forward_net(cat2)
        return a_hat, s_hat

    def predict_once(self, s, a, s_prime):
        '''Predicts next action and state for a single observation.'''
        s = torch.from_numpy(s).float()
        s_prime = torch.from_numpy(s_prime).float()
        cat1 = torch.cat([self.feature_net(s), self.feature_net(s_prime)])
        cat2 = torch.cat([self.feature_net(s).clone().detach(), a])
        a_hat = self.inverse_net(cat1)
        s_hat = self.forward_net(cat2)
        return a_hat, s_hat

    def loss(self, s, a, a_hat, s_prime, s_hat):
        '''Calculates loss.'''
        L_I = self.L_I(a_hat, a.view(-1,self.action_size))
        L_F = self.L_F(s_hat, self.feature_net(s_prime.view(-1,self.state_size)).clone().detach())
        return (1 - self.β) * L_I + self.β * L_F

    def loss_once(self, s, a, a_hat, s_prime, s_hat):
        '''Calculates loss for a single prediction.'''
        s_prime = torch.from_numpy(s_prime).float()
        L_I = self.L_I(a_hat, a)
        L_F = self.L_F(s_hat, self.feature_net(s_prime).clone().detach())
        return (1 - self.β) * L_I + self.β * L_F
            