#!/usr/bin/bash
./scripts/run_ir_experiment.sh re3 '"obs_shape": "env_observation_shape", "action_shape": "env_action_shape", "latent_dim": 128, "beta": 0.01, "kappa": 0.001' batch_ir.yml --experiment-suffix _re3 $@
./scripts/run_ir_experiment.sh rise '"obs_shape": "env_observation_shape", "action_shape": "env_action_shape", "latent_dim": 128, "beta": 0.01, "kappa": 0.001' batch_ir.yml --experiment-suffix _rise $@
./scripts/run_ir_experiment.sh ride '"obs_shape": "env_observation_shape", "action_shape": "env_action_shape", "latent_dim": 128, "beta": 0.01, "kappa": 0.001' batch_ir.yml --experiment-suffix _ride $@
./scripts/run_ir_experiment.sh revd '"obs_shape": "env_observation_shape", "action_shape": "env_action_shape", "latent_dim": 128, "beta": 0.01, "kappa": 0.001' batch_ir.yml --experiment-suffix _revd $@
./scripts/run_ir_experiment.sh rnd '"obs_shape": "env_observation_shape", "action_shape": "env_action_shape", "latent_dim": 128, "lr": 0.001, "batch_size": 32, "beta": 0.01, "kappa": 0.001' batch_ir.yml --experiment-suffix _rnd $@
./scripts/run_ir_experiment.sh ngu '"envs": "envs", "latent_dim": 128, "lr": 0.001, "batch_size": 32, "beta": 0.01, "kappa": 0.001' batch_ir.yml --experiment-suffix _ngu $@
./scripts/run_ir_experiment.sh icm '"envs": "envs", "lr": 0.001, "batch_size": 32, "beta": 0.01, "kappa": 0.001' batch_ir.yml --experiment-suffix _icm $@
./scripts/run_ir_experiment.sh girm '"envs": "envs", "latent_dim": 128, "lr": 0.001, "batch_size": 32, "lambd": 0.1, "beta": 0.01, "kappa": 0.001' batch_ir.yml --experiment-suffix _girm $@
