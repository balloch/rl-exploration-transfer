#!/bin/bash
./scripts/run_sweep.sh cartpole/cartpole_sweeps/re3.yml cartpole/cartpole_sweeps/base.yml re3.yml ir_ppo.yml cartpole/tuned/ppo.yml $@
./scripts/run_sweep.sh cartpole/cartpole_sweeps/rise.yml cartpole/cartpole_sweeps/base.yml rise.yml ir_ppo.yml cartpole/tuned/ppo.yml $@
./scripts/run_sweep.sh cartpole/cartpole_sweeps/ride.yml cartpole/cartpole_sweeps/base.yml ride.yml ir_ppo.yml cartpole/tuned/ppo.yml $@
./scripts/run_sweep.sh cartpole/cartpole_sweeps/revd.yml cartpole/cartpole_sweeps/base.yml revd.yml ir_ppo.yml cartpole/tuned/ppo.yml $@
./scripts/run_sweep.sh cartpole/cartpole_sweeps/rnd.yml cartpole/cartpole_sweeps/base.yml rnd.yml ir_ppo.yml cartpole/tuned/ppo.yml $@
./scripts/run_sweep.sh cartpole/cartpole_sweeps/ngu.yml cartpole/cartpole_sweeps/base.yml ngu.yml ir_ppo.yml cartpole/tuned/ppo.yml $@
./scripts/run_sweep.sh cartpole/cartpole_sweeps/icm.yml cartpole/cartpole_sweeps/base.yml icm.yml ir_ppo.yml cartpole/tuned/ppo.yml $@
./scripts/run_sweep.sh cartpole/cartpole_sweeps/girm.yml cartpole/cartpole_sweeps/base.yml girm.yml ir_ppo.yml cartpole/tuned/ppo.yml $@
./scripts/run_sweep.sh cartpole/cartpole_sweeps/diayn.yml cartpole/cartpole_sweeps/base.yml diayn.yml ir_ppo.yml cartpole/tuned/ppo.yml $@
