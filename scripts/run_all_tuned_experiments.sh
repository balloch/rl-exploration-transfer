#!/usr/bin/bash
./scripts/run_experiment.sh tuned/ppo.yml base.yml $@ --experiment-suffix _none
./scripts/run_experiment.sh tuned/re3.yml ir_ppo.yml tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh tuned/rise.yml ir_ppo.yml tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh tuned/ride.yml ir_ppo.yml tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh tuned/revd.yml ir_ppo.yml tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh tuned/rnd.yml ir_ppo.yml tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh tuned/ngu.yml ir_ppo.yml tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh tuned/icm.yml ir_ppo.yml tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh tuned/girm.yml ir_ppo.yml tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh tuned/diayn.yml ir_ppo.yml tuned/ppo.yml base.yml $@
./scripts/run_experiment.sh tuned/noisy_nets_ac.yml tuned/ppo.yml base.yml $@