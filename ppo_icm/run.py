import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from ppo import PPO
from icm import ICM

# Settings
env_name = 'MountainCarContinuous-v0'
render = False
load_model = False
load_model_filename = 'PPO_MountainCarContinuous-v0_5000.pth'
use_icm = True
env_seed = 182

# Hyperparameters
learning_rate  = 0.002
gamma          = 0.9
lmbda          = 0.9
eps_clip       = 0.2 
K_epoch        = 10
rollout_len    = 3
buffer_size    = 30
minibatch_size = 32
policy_weight = 1.0
max_episodes = 300

# ICM parameters
feature_hidden_sizes = [128,128,]
feature_size = 8
inverse_hidden_sizes = [128,128,]
forward_hidden_sizes = [128,128,]
β = 0.5

def main():
    # Create gym environment
    env = gym.make(env_name)

    # Set seed for reproducibility
    env.seed(env_seed)
    env.action_space.seed(env_seed)

    # Model initialization
    model = PPO(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        learning_rate=learning_rate,
        gamma = gamma,
        lmbda = lmbda,
        eps_clip = eps_clip,
        K_epoch = K_epoch,
        buffer_size = buffer_size,
        minibatch_size = minibatch_size,
        policy_weight = policy_weight,
        use_icm = use_icm,
        icm_parameters = (feature_hidden_sizes, feature_size, inverse_hidden_sizes, forward_hidden_sizes, β)
    )

    # For loading previous models
    if load_model:
        model.load_state_dict(torch.load('./saved_models/' + load_model_filename))

    # Auxiliary parameters
    score = 0.0
    print_interval = 10
    save_interval = 100
    rollout = []
    avg_scores = []
    first_time = True
    first_avg_score = 0

    # Train over episodes
    for n_epi in range(max_episodes):
        s = env.reset()
        env.seed(env_seed)
        env.action_space.seed(env_seed)
        done = False
        while not done:
            for t in range(rollout_len):
                # Distribution to select action
                mu, std = model.pi(torch.from_numpy(s).float())
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)
                s_prime, r, done, info = env.step(a)
                
                if render:
                    env.render()
                
                # Appending to training data
                rollout.append((s, a, r/10.0, s_prime, log_prob, done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                s = s_prime
                score += r
                if done:
                    break

            # Train model
            model.train_net()

        # Record score
        if n_epi%print_interval==0 and n_epi!=0:
            avg_score = score/print_interval
            avg_scores.append(avg_score)
            print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, avg_score, model.optimization_step))
            score = 0.0
            if first_time:
                first_time = False
                first_avg_score = avg_score

        # Save model itself
        if n_epi%save_interval==0 and n_epi!=0:
            if use_icm:
                torch.save(model.state_dict(), './saved_models/PPO_ICM_{}_{}.pth'.format(env_name, n_epi))
            else:
                torch.save(model.state_dict(), './saved_models/PPO_{}_{}.pth'.format(env_name, n_epi))

    env.close()

    # Save model results
    if use_icm:
        np.save('./results_seeded/PPO_ICM_{}_{}_{}_{}'.format(env_name, first_avg_score, policy_weight, β), np.array(avg_scores))
    else:
        np.save('./results_seeded/PPO_{}_{}'.format(env_name, first_avg_score), np.array(avg_scores))

main()