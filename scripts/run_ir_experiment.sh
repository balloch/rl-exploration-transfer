#!/usr/bin/bash
./scripts/run_experiment.sh ir_ppo_base.yml ${@:3} --rl-alg-kwargs '{"ir_alg_cls": '"\"$1\""', "ir_alg_kwargs": {'"$2"'}}'