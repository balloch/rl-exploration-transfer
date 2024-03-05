#!/bin/bash
./scripts/run_sweep.sh sweeps/ppo.yml sweeps/base.yml ppo.yml $@ --experiment-suffix _none
./scripts/run_sweep.sh sweeps/re3.yml sweeps/base.yml re3.yml ir_ppo.yml ppo.yml $@
./scripts/run_sweep.sh sweeps/rise.yml sweeps/base.yml rise.yml ir_ppo.yml ppo.yml $@
./scripts/run_sweep.sh sweeps/ride.yml sweeps/base.yml ride.yml ir_ppo.yml ppo.yml $@
./scripts/run_sweep.sh sweeps/revd.yml sweeps/base.yml revd.yml ir_ppo.yml ppo.yml $@
./scripts/run_sweep.sh sweeps/rnd.yml sweeps/base.yml rnd.yml ir_ppo.yml ppo.yml $@
./scripts/run_sweep.sh sweeps/ngu.yml sweeps/base.yml ngu.yml ir_ppo.yml ppo.yml $@
./scripts/run_sweep.sh sweeps/icm.yml sweeps/base.yml icm.yml ir_ppo.yml ppo.yml $@
./scripts/run_sweep.sh sweeps/girm.yml sweeps/base.yml girm.yml ir_ppo.yml ppo.yml $@
./scripts/run_sweep.sh sweeps/diayn.yml sweeps/base.yml diayn.yml ir_ppo.yml ppo.yml $@
./scripts/run_sweep.sh sweeps/noisy_nets.yml sweeps/base.yml noisy_nets_ac.yml ppo.yml $@
