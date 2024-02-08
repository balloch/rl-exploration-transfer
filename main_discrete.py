import gym
import minigrid
import gymnasium
from Brain import SACAgent,SACAgentDiscrete
from Common import Play_Discrete, Logger, get_params
import numpy as np
from tqdm import tqdm
#import mujoco_py


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    params = get_params()

    #test_env = gym.make(params["env_name"])
    env = gymnasium.make(params["env_name"],render_mode='rgb_array')

    print(params['env_name'])
    
    #env = gym.make('MiniGrid-DoorKey-6x6-v0')
    env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
    #env = minigrid.wrappers.FlatObsWrapper(env)
    obs, _ = env.reset()
    print(obs['image'].flatten())
    obs = obs['image'].flatten()
    n_states = obs.shape[0]
    ACTION_SPACE_LIST = [i for i in range(env.action_space)]
    ACTION_SPACE= len(ACTION_SPACE_LIST)
    n_actions = 1 #ACTION_SPACE
    action_bounds = [ACTION_SPACE_LIST[0],ACTION_SPACE_LIST[-1]]#[0,2] # [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    env.close()
    #del env, n_states, n_actions

    #env = gym.make(params["env_name"])

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgentDiscrete(p_z=p_z, **params)
    logger = Logger(agent, **params)
    print(params)
    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            env.np_random.set_state(env_rng_states[0])
            env.observation_space.np_random.set_state(env_rng_states[1])
            env.action_space.np_random.set_state(env_rng_states[2])
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            print(params['seed'])
            np.random.seed(params["seed"])
            env.reset(seed=params["seed"])
            #env.observation_space.seed(params["seed"])
            #env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state, _ = env.reset()
            state = state['image'].flatten()
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            max_n_steps = 360 # min(params["max_episode_len"], env.spec.max_episode_steps)
            for step in range(1, 1 + max_n_steps):

                action =  round(agent.choose_action(state)[0])#agent.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state['image'].flatten()
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state)
                logq_zs = agent.train()
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
                if done:
                    break

            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       np.random.get_state(),
                       #env.np_random.get_state(),
                       #env.observation_space.np_random.get_state(),
                       #env.action_space.np_random.get_state(),
                       #*agent.get_rng_states()
                       )

    else:
        #logger.load_weights()
        player = Play_Discrete(env, agent, n_skills=params["n_skills"])
        player.evaluate()
