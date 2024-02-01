# Intrinsic Reward NovGrid Runner
An experiment runner script for intrinsic reward exploration algorithms running on environments with transfers embedding in the training.
## Run Command
From the root of this repo, use the following command to run an experiment:
```bash
python experiments/ir_novgrid.py -c {name of config file}
```
For example:
```bash
python experiments/ir_novgrid.py -c defaults
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
]```
This env config will have the agent start in the `MiniGrid-SimpleCrossing` environment, then transfer to `MiniGrid-LavaCrossing`, and then back.
Further, different environment specifications can be specified within this json, changing the size of environments or other settings.


## Arguments
### Reference Table
|Short     |Long           |Config File Key|Default             |Type           |Help                                                                                                                    |
|----------|---------------|---------------|--------------------|---------------|------------------------------------------------------------------------------------------------------------------------
|-h        |--help         |help           |==SUPPRESS==        |None           |show this help message and exit                                                                                         |
|--env-config-file|-ec            |env_config_file|simple_to_lava_crossing.json|str            |Use the path to a json file containing the env configs here.                                                            |
|--total-time-steps|-t             |total_time_steps|10000000            |int            |The total number of time steps to run.                                                                                  |
|--novelty-step|-n             |novelty_step   |3000000             |int            |The total number of time steps to run in an environment before injecting the next novelty.                              |
|--n-envs  |-e             |n_envs         |5                   |int            |The number of envs to use when running the vectorized env.                                                              |
|--experiment-name|-en            |experiment_name|novgrid_simple_to_lava_ppo_re3|str            |The name of the experiment.                                                                                             |
|--sb3-model|-m             |sb3_model      |PPO                 |str            |The name of the stable baselines model to use. Examples include PPO, DQN, etc.                                          |
|--policy  |-p             |policy         |MlpPolicy           |str            |The type of policy to use. Examples include MlpPolicy, CnnPolicy, etc.                                                  |
|--ir-alg  |-ir            |ir_alg         |RE3                 |str            |The intrinsic reward algorithm to use. Examples include RE3, RND, NGU, etc.                                             |
|--ir-beta |-irb           |ir_beta        |0.01                |float          |The beta parameter for the intrinsic reward algorithm.                                                                  |
|--ir-kappa|-irk           |ir_kappa       |1e-05               |float          |The kappa parameter for the intrinsic reward algorithm.                                                                 |
|--ir-learning-rate|-irl           |ir_learning_rate|0.001               |float          |The learning rate parameter for the intrinsic reward algorithm.                                                         |
|--ir-latent-dim|-irld          |ir_latent_dim  |128                 |int            |The latent dim parameter for the intrinsic reward algorithm.                                                            |
|--ir-batch-size|-irbs          |ir_batch_size  |32                  |int            |The batch size parameter for the intrinsic reward algorithm.                                                            |
|--ir-lambda|-irlb          |ir_lambda      |0.1                 |float          |The lambda parameter for the intrinsic reward algorithm.                                                                |
|--wandb-project-name|-wpn           |wandb_project_name|rl-transfer-explore |str            |The project name to save under in wandb.                                                                                |
|--wandb-save-videos|-wsv           |wandb_save_videos|False               |bool           |Whether or not to save videos to wandb.                                                                                 |
|--wandb-video-freq|-wvf           |wandb_video_freq|2000                |int            |How often to save videos to wandb.                                                                                      |
|--wandb-video-length|-wvl           |wandb_video_length|200                 |int            |How long the videos saved to wandb should be                                                                            |
|--wandb-model-save-freq|-wmsf          |wandb_model_save_freq|100000              |int            |How often to save the model.                                                                                            |
|--wandb-gradient-save-freq|-wgsf          |wandb_gradient_save_freq|0                   |int            |How often to save the gradients.                                                                                        |
|--wandb-verbose|-wv            |wandb_verbose  |2                   |int            |The verbosity setting for wandb.                                                                                        |
|--n-runs  |-r             |n_runs         |5                   |int            |The number of runs to do.                                                                                               |
|--log     |-l             |log            |True                |bool           |Whether or not to log the results to tensor board and wandb.                                                            |
|--save-model|-s             |save_model     |True                |bool           |Whether or not to save the model if wandb didn't already.                                                               |
|--log-interval|-li            |log_interval   |1                   |int            |The log interval for model.learn.                                                                                       |
|--print-novelty-box|-pnb           |print_novelty_box|True                |bool           |Whether or not to print the novelty box when novelty occurs.                                                            |
|--verbose |-v             |verbose        |1                   |int            |The verbosity parameter for model.learn.                                                                                |
|-c        |--config-file  |config_file    |[]                  |str            |A config file specifying some new default arguments (that can be overridden by command line args). This can be a list of config files (json/yml/yaml) with whatever arguments are set in them in order of lowest to highest priority. Usage: --config-file test.json test2.json.|



