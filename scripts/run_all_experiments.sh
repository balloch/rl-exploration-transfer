#!/usr/bin/bash
./scripts/run_experiment.sh ppo.yml base.yml $@ --experiment-suffix _none
./scripts/run_experiment.sh re3.yml ir_ppo.yml base.yml $@
./scripts/run_experiment.sh rise.yml ir_ppo.yml base.yml $@
./scripts/run_experiment.sh ride.yml ir_ppo.yml base.yml $@
./scripts/run_experiment.sh revd.yml ir_ppo.yml base.yml $@
./scripts/run_experiment.sh rnd.yml ir_ppo.yml base.yml $@
./scripts/run_experiment.sh ngu.yml ir_ppo.yml base.yml $@
./scripts/run_experiment.sh icm.yml ir_ppo.yml base.yml $@
./scripts/run_experiment.sh girm.yml ir_ppo.yml base.yml $@
./scripts/run_experiment.sh diayn.yml ir_ppo.yml base.yml $@
./scripts/run_experiment.sh noisy_nets_ac.yml ppo.yml base.yml $@