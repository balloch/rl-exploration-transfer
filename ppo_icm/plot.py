import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------

# Plots  for PPO vs. PPO with ICM

runs = [
    'PPO_ICM_MountainCarContinuous-v0_-37.07607583799019.npy',
    'PPO_ICM_MountainCarContinuous-v0_-46.06291450960883_0.75_0.2.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.624374026419844_1.0_0.2.npy',
    'PPO_ICM_MountainCarContinuous-v0_-97.24413804480204.npy',
    'PPO_MountainCarContinuous-v0_-71.65956278338766.npy',
    'PPO_MountainCarContinuous-v0_-74.31046497716287.npy',
    'PPO_MountainCarContinuous-v0_-110.17387717608476.npy'
]

directory = './results/'
savefile = './plots/ppo_vs_ppo_icm'
loop_runs = runs

plot_range = 30
ppo_count = 0
ppo_icm_count = 0

for run in loop_runs:
    y = np.load(directory+run)
    y = y[:plot_range]
    if run.startswith('PPO_ICM'):
        color = 'b'
        label= 'PPO_ICM'
        ppo_icm_count += 1
    else:
        color = 'r'
        label='PPO'
        ppo_count += 1
    if ppo_count == 1 or ppo_icm_count == 1:
        plt.plot(np.arange(len(y)) * 10, y, color, label=label)
    else:
        plt.plot(np.arange(len(y)) * 10, y, color)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('MountainCarContinuous Score')
plt.legend()
plt.savefig(savefile)
plt.show()

# --------------------------------------------------------------------------

# Plots for PPO vs. PPO with ICM with seed set

seed_runs = [
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.5.npy',
    'PPO_MountainCarContinuous-v0_-50.27348071147965.npy'
]

directory = './results_seeded/'
savefile = './plots/ppo_vs_ppo_icm_seeded'
loop_runs = seed_runs

for run in loop_runs:
    y = np.load(directory+run)
    y = y[:plot_range]
    if run.startswith('PPO_ICM'):
        plt.plot(np.arange(len(y)) * 10, y, label='PPO_ICM')
    else:
        plt.plot(np.arange(len(y)) * 10, y, label='PPO')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('MountainCarContinuous Score (Seeded)')
plt.legend()
plt.savefig(savefile)
plt.show()

# --------------------------------------------------------------------------

# PPO with ICM for varying β

seed_runs = [
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.0.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.1.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.2.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.5.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.75.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.9.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_1.0.npy',
]

savefile = './plots/ppo_vs_ppo_icm_seeded_betas'
loop_runs = seed_runs

beta = [0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0]

for i, run in enumerate(loop_runs):
    y = np.load(directory+run)
    y = y[:plot_range]
    plt.plot(np.arange(len(y)) * 10, y, label='β='+str(beta[i]))
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('MountainCarContinuous Score, Vary β (Seeded, λ=1.0)')
plt.legend()
plt.savefig(savefile)
plt.show()

# --------------------------------------------------------------------------

# PPO with ICM for varying λ

seed_runs = [
    'PPO_ICM_MountainCarContinuous-v0_-50.71670728565125_0.0_0.5.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.12280145402665_0.01_0.5.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.11093076309284_0.1_0.5.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10932288555409_0.5_0.5.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.5.npy'
]

savefile = './plots/ppo_vs_ppo_icm_seeded_lmbdas'
loop_runs = seed_runs

lmbda = [0.0, 0.01, 0.1, 0.5, 1.0]

for i, run in enumerate(loop_runs):
    y = np.load(directory+run)
    y = y[:plot_range]
    plt.plot(np.arange(len(y)) * 10, y, label='λ='+str(lmbda[i]), alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('MountainCarContinuous Score, Vary λ (Seeded, β=0.5)')
plt.legend()
plt.savefig(savefile)
plt.show()