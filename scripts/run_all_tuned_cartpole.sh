#!/bin/bash
./scripts/run_experiment.sh cartpole/tuned/ppo.yml base.yml $@ --experiment-suffix _none
./scripts/run_experiment.sh cartpole/tuned/re3.yml ir_ppo.yml cartpole/tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh cartpole/tuned/rise.yml ir_ppo.yml cartpole/tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh cartpole/tuned/ride.yml ir_ppo.yml cartpole/tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh cartpole/tuned/revd.yml ir_ppo.yml cartpole/tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh cartpole/tuned/rnd.yml ir_ppo.yml cartpole/tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh cartpole/tuned/ngu.yml ir_ppo.yml cartpole/tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh cartpole/tuned/icm.yml ir_ppo.yml cartpole/tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh cartpole/tuned/girm.yml ir_ppo.yml cartpole/tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh cartpole/tuned/diayn.yml ir_ppo.yml cartpole/tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh cartpole/tuned/noisy_nets_ac.yml cartpole/tuned/ppo.yml base.yml $@
