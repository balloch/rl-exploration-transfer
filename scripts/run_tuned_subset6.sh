#!/bin/bash
methods=$(echo $1 | tr "," "\n")
for var in $methods
do
    command=$(cat scripts/run_all_tuned_experiments.sh | grep "$var")
    command="${command//\$@/"${@:2} --n-runs 5 --total-time-steps 3000000 --novelty-step 1000000 --env-configs-file walker_friction --wrappers FlattenObservation"}" 
    eval "$command"
done