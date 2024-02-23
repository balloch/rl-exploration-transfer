# Main Experiment Runner
An experiment runner script for intrinsic reward exploration algorithms running on environments with transfers embedding in the training.
## Run Command
From the root of this repo, use the following command to run an experiment:
```bash
python experiments/main.py -c {name of config file(s)} {additional config here}
```
For example:
```bash
python experiments/main.py -c ppo.yml base.yml
```



## Run Scripts
From the root of this repo, these commands will also start experiment runs.


To run a single experiment:
```bash
./scripts/run_experiment.sh {name of config file(s)} {additional config here}
```
To run all the preset experiments using all the different exploration methods:
```bash
./scripts/run_all_experiments.sh {additional config here}
```
To run a subset of the preset experiments:
```bash
./scripts/run_subset.sh {selector string} {additional config here}
```
Example usage of the subset script is as follows:
```bash
./scripts/run_subset.sh girm,diayn,re3 debug.yml --total-time-steps 10000000
```



## Config Files
When specifying a config file, the codebase will look in `configs` for the config file. The config file can be left blank as well and the defaults listed below will be used. Further, arguments can be overridden via the command line. 



## Env Configs
These are environment config files (and can be specified in the main config file or via the command line) that specify the different environments/transfers that the agent must traverse. For example, the `simple_to_lava_crossing.json` file looks like:
```json
[
	{
		"env_id": "MiniGrid-SimpleCrossingS9N3-v0"
	},
	{
		"env_id": "MiniGrid-LavaCrossingS9N2-v0"
	},
	{
		"env_id": "MiniGrid-SimpleCrossingS9N3-v0"
	}
]
```
This env config will have the agent start in the `MiniGrid-SimpleCrossing` environment, then transfer to `MiniGrid-LavaCrossing`, and then back.
Further, different environment specifications can be specified within this json, changing the size of environments or other settings.


## Arguments
### Reference Table
|Short     |Long           |Config File Key|Default             |Type           |Help                                                                                                                    |
|----------|---------------|---------------|--------------------|---------------|------------------------------------------------------------------------------------------------------------------------
|-h        |--help         |help           |==SUPPRESS==        |None           |show this help message and exit                                                                                         |
|--env-configs-file|-ec            |env_configs_file|simple_to_lava_to_simple_crossing|str            |Use the path to a json file containing the env configs here.                                                            |
|--total-time-steps|-t             |total_time_steps|10000000            |int            |The total number of time steps to run.                                                                                  |
|--novelty-step|-n             |novelty_step   |3000000             |int            |The total number of time steps to run in an environment before injecting the next novelty.                              |
|--n-envs  |-e             |n_envs         |5                   |int            |The number of envs to use when running the vectorized env.                                                              |
|--experiment-name|-en            |experiment_name|None                |str            |The name of the experiment.                                                                                             |
|--experiment-prefix|-ep            |experiment_prefix|novgrid_            |str            |The prefix for the experiment name to use when the experiment name is not explicitly defined.                           |
|--experiment-suffix|-es            |experiment_suffix|                    |str            |The suffix for the experiment name to use when the experiment name is not explicitly defined.                           |
|--rl-alg  |-a             |rl_alg         |PPO                 |rlexplore.\*/stable_baselines3.\*/BaseAlgorithm.\*|The name of the stable baselines model to use. Examples include PPO, DQN, etc.                                          |
|--rl-alg-kwargs|-ak            |rl_alg_kwargs  |{}                  |json           |The kwargs to pass to the RL algorithm. These include the intrinsic reward class name and kwargs if using an IR model.  |
|--policy  |-p             |policy         |MlpPolicy           |rlexplore.\*/stable_baselines3.common.policies.\*|The type of policy to use. Examples include MlpPolicy, CnnPolicy, etc.                                                  |
|--policy-kwargs|-pk            |policy_kwargs  |{}                  |json           |The kwargs to pass to the policy.                                                                                       |
|--wrappers|-w             |wrappers       |[<class 'minigrid.wrappers.ImgObsWrapper'>, <class 'gymnasium.wrappers.flatten_observation.FlattenObservation'>]|rlexplore.\*/minigrid.wrappers.\*/gymnasium.wrappers.\*|The wrappers to use on the environment.                                                                                 |
|--wrappers-kwargs|-wk            |wrappers_kwargs|[]                  |json           |The arguments for the wrappers to use on the environment.                                                               |
|--callbacks|-cb            |callbacks      |[]                  |rlexplore.\*/stable_baselines3.common.callbacks.\*|The callbacks to pass to the model.learn function.                                                                      |
|--callbacks-kwargs|-cbk           |callbacks_kwargs|[]                  |json           |THe arguments for the callbacks to use on the model.learn call.                                                         |
|--wandb-project-name|-wpn           |wandb_project_name|rl-transfer-explore |str            |The project name to save under in wandb.                                                                                |
|--wandb-save-videos|-wsv           |wandb_save_videos|False               |bool           |Whether or not to save videos to wandb.                                                                                 |
|--wandb-video-freq|-wvf           |wandb_video_freq|2000                |int            |How often to save videos to wandb.                                                                                      |
|--wandb-video-length|-wvl           |wandb_video_length|200                 |int            |How long the videos saved to wandb should be                                                                            |
|--wandb-model-save-freq|-wmsf          |wandb_model_save_freq|100000              |int            |How often to save the model.                                                                                            |
|--wandb-gradient-save-freq|-wgsf          |wandb_gradient_save_freq|0                   |int            |How often to save the gradients.                                                                                        |
|--wandb-verbose|-wv            |wandb_verbose  |2                   |int            |The verbosity setting for wandb.                                                                                        |
|--n-runs  |-r             |n_runs         |10                  |int            |The number of runs to do.                                                                                               |
|--log     |-l             |log            |True                |bool           |Whether or not to log the results to tensor board and wandb.                                                            |
|--save-model|-s             |save_model     |True                |bool           |Whether or not to save the model if wandb didn't already.                                                               |
|--log-interval|-li            |log_interval   |1                   |int            |The log interval for model.learn.                                                                                       |
|--print-novelty-box|-pnb           |print_novelty_box|True                |bool           |Whether or not to print the novelty box when novelty occurs.                                                            |
|--verbose |-v             |verbose        |1                   |int            |The verbosity parameter for model.learn.                                                                                |
|--device  |-d             |device         |cuda:0              |str            |The torch device string to use.                                                                                         |
|--gpu-idx |-gi            |gpu_idx        |None                |str            |The gpu index to use.                                                                                                   |
|-c        |--config-file  |config_file    |[]                  |str            |A config file specifying some new default arguments (that can be overridden by command line args). This can be a list of config files (json/yml/yaml) with whatever arguments are set in them in order of lowest to highest priority. Usage: --config-file test.json test2.json.|



# Hyperparameter Sweep
A hyperparameter tuning script that uses wandb sweeps to tune the specified hyperparameters against the convergence speed metric.
## Run Command
From the root of this repo, use the following command to run an experiment:
```bash
python experiments/sweep.py -c {name of sweep config file(s)} {additional config here}
```
For example:
```bash
python experiments/sweep.py -c sweeps/ppo.yml sweeps/base.yml ppo.yml
```



## Run Scripts
From the root of this repo, these commands will also start sweeps.


To run a single sweep:
```bash
./scripts/run_sweep.sh {name of sweep config file(s)} {additional config here}
```
To run all the preset sweeps using all the different exploration methods:
```bash
./scripts/run_all_sweeps.sh {additional config here}
```
To run a subset of the preset sweeps:
```bash
./scripts/run_subset_sweeps.sh {selector string} {additional config here}
```
Example usage of the subset script is as follows:
```bash
./scripts/run_subset_sweeps.sh girm,diayn,re3 debug.yml
```





## Arguments
### Reference Table
|Short     |Long           |Config File Key|Default             |Type           |Help                                                                                                                    |
|----------|---------------|---------------|--------------------|---------------|------------------------------------------------------------------------------------------------------------------------
|-h        |--help         |help           |==SUPPRESS==        |None           |show this help message and exit                                                                                         |
|--env-configs-file|-ec            |env_configs_file|simple_to_lava_to_simple_crossing|str            |Use the path to a json file containing the env configs here.                                                            |
|--total-time-steps|-t             |total_time_steps|10000000            |int            |The total number of time steps to run.                                                                                  |
|--n-envs  |-e             |n_envs         |5                   |int            |The number of envs to use when running the vectorized env.                                                              |
|--experiment-name|-en            |experiment_name|None                |str            |The name of the experiment.                                                                                             |
|--experiment-prefix|-ep            |experiment_prefix|novgrid_            |str            |The prefix for the experiment name to use when the experiment name is not explicitly defined.                           |
|--experiment-suffix|-es            |experiment_suffix|                    |str            |The suffix for the experiment name to use when the experiment name is not explicitly defined.                           |
|--rl-alg  |-a             |rl_alg         |PPO                 |rlexplore.\*/stable_baselines3.\*/BaseAlgorithm.\*|The name of the stable baselines model to use. Examples include PPO, DQN, etc.                                          |
|--rl-alg-kwargs|-ak            |rl_alg_kwargs  |{'learning_rate': '$learning_rate'}|json           |The kwargs to pass to the RL algorithm. These include the intrinsic reward class name and kwargs if using an IR model.  |
|--policy  |-p             |policy         |MlpPolicy           |rlexplore.\*/stable_baselines3.common.policies.\*|The type of policy to use. Examples include MlpPolicy, CnnPolicy, etc.                                                  |
|--policy-kwargs|-pk            |policy_kwargs  |{}                  |json           |The kwargs to pass to the policy.                                                                                       |
|--wrappers|-w             |wrappers       |[<class 'minigrid.wrappers.ImgObsWrapper'>, <class 'gymnasium.wrappers.flatten_observation.FlattenObservation'>]|rlexplore.\*/minigrid.wrappers.\*/gymnasium.wrappers.\*|The wrappers to use on the environment.                                                                                 |
|--wrappers-kwargs|-wk            |wrappers_kwargs|[]                  |json           |The arguments for the wrappers to use on the environment.                                                               |
|--wandb-project-name|-wpn           |wandb_project_name|rl-transfer-explore-sweeps|str            |The project name to save under in wandb.                                                                                |
|--n-runs  |-r             |n_runs         |10                  |int            |The number of runs to do.                                                                                               |
|--log-interval|-li            |log_interval   |1                   |int            |The log interval for model.learn.                                                                                       |
|--print-novelty-box|-pnb           |print_novelty_box|True                |bool           |Whether or not to print the novelty box when novelty occurs.                                                            |
|--verbose |-v             |verbose        |1                   |int            |The verbosity parameter for model.learn.                                                                                |
|--device  |-d             |device         |cuda:0              |str            |The torch device string to use.                                                                                         |
|--gpu-idx |-gi            |gpu_idx        |None                |str            |The gpu index to use.                                                                                                   |
|--sweep-env-configs|-secf          |sweep_env_configs|[]                  |json           |Use the path to a json file containing the env configs here.                                                            |
|--sweep-configuration|-sc            |sweep_configuration|{}                  |json           |The sweep configuration to use in wandb.                                                                                |
|--eval-freq|-ef            |eval_freq      |10000               |int            |The frequency to evaluate the agent.                                                                                    |
|--eval-episodes|-ee            |eval_episodes  |10                  |int            |The number of eval episodes to use.                                                                                     |
|--min-reward-threshold|-mrt           |min_reward_threshold|0.75                |float          |The min reward threshold to stop training in the eval callback.                                                         |
|-c        |--config-file  |config_file    |[]                  |str            |A config file specifying some new default arguments (that can be overridden by command line args). This can be a list of config files (json/yml/yaml) with whatever arguments are set in them in order of lowest to highest priority. Usage: --config-file test.json test2.json.|



# Tag Wandb Runs
A python script to tag all the wandb runs with tags that allow for easier filtering.
## Run Command
From the root of this repo, use the following command to tag all wandb runs:
```bash
python experiments/tag_wandb_runs.py {argparse config here}





## Arguments
### Reference Table
|Short     |Long           |Config File Key|Default             |Type           |Help                                                                                                                    |
|----------|---------------|---------------|--------------------|---------------|------------------------------------------------------------------------------------------------------------------------
|-h        |--help         |help           |==SUPPRESS==        |None           |show this help message and exit                                                                                         |
|--wandb-project-name|-wpn           |wandb_project_name|rl-transfer-explore |str            |The project name to load from in wandb.                                                                                 |
|--convergence-reward-threshold|-crt           |convergence_reward_threshold|0.8                 |float          |The convergence threshold on the reward.                                                                                |
|--convergence-check-step-ratio|-ccsr          |convergence_check_step_ratio|0.9                 |float          |The convergence step ratio to check on each environment. If the ratio is 0.9 and the environment was trained on for 10 steps, the code will check at step 9.|
|-c        |--config-file  |config_file    |[]                  |str            |A config file specifying some new default arguments (that can be overridden by command line args). This can be a list of config files (json/yml/yaml) with whatever arguments are set in them in order of lowest to highest priority. Usage: --config-file test.json test2.json.|



# Plot Data
A python script to pull the reward data from wandb and plot it using seaborn.
## Run Command
From the root of this repo, use the following command to plot the reward data:
```bash
python experiments/plot_data.py {argparse config here}





## Arguments
### Reference Table
|Short     |Long           |Config File Key|Default             |Type           |Help                                                                                                                    |
|----------|---------------|---------------|--------------------|---------------|------------------------------------------------------------------------------------------------------------------------
|-h        |--help         |help           |==SUPPRESS==        |None           |show this help message and exit                                                                                         |
|--wandb-project-name|-wpn           |wandb_project_name|rl-transfer-explore |str            |The project name to load from in wandb.                                                                                 |
|--env-configs-file|-ec            |env_configs_file|door_key_change     |str            |The env configs file name used in the experiments to plot.                                                              |
|--n-tasks |-n             |n_tasks        |2                   |int            |The number of tasks run in these experiments. Should correspond with env configs file.                                  |
|--filter-unconverged-out|-fuo           |filter_unconverged_out|True                |bool           |Whether or not to filter our the unconverged runs                                                                       |
|--img-name|-i             |img_name       |converged_ep_rew_mean.png|str            |The name of the image to save the plot to in the figures folder.                                                        |
|--estimator|-e             |estimator      |mean                |estimator_type |The estimator to use (in seaborns lineplot function) to aggregate data.                                                 |
|--error-bar-type|-ebt           |error_bar_type |ci                  |str            |The type of error bar (in seaborns lineplot function) to use.                                                           |
|--error-bar-arg|-eba           |error_bar_arg  |95                  |float          |The type of error bar argument (in seaborns lineplot function) to use.                                                  |
|--step-range|-sr            |step_range     |(0, 0)              |int_tuple      |The range of steps to include in the data.                                                                              |
|-c        |--config-file  |config_file    |[]                  |str            |A config file specifying some new default arguments (that can be overridden by command line args). This can be a list of config files (json/yml/yaml) with whatever arguments are set in them in order of lowest to highest priority. Usage: --config-file test.json test2.json.|



