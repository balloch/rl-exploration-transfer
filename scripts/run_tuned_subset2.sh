#!/bin/bash
methods=$(echo $1 | tr "," "\n")
for var in $methods
do
    command=$(cat scripts/run_all_tuned_experiments.sh | grep "$var")
    command="${command//\$@/"${@:2} --total-time-steps 5000000 --novelty-step 2000000 --env-configs-file simple_to_lava_crossing"}" 
    eval "$command"
done