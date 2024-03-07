#!/bin/bash
methods=$(echo $1 | tr "," "\n")
for var in $methods
do
    command=$(cat scripts/run_all_tuned_cartpole.sh | grep "$var")
    command="${command//\$@/"${@:2} --n-runs 5 --n-envs 10 --novelty-step 10000000 --total-time-steps 20000000 --env-configs-file walker_thigh_length --wrappers FlattenObservation"}" 
    eval "$command"
done