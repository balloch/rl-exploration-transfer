#!/usr/bin/bash
methods=$(echo $1 | tr "," "\n")
for var in $methods
do
    command=$(cat scripts/run_all_experiments.sh | grep "$var")
    command="${command//\$@/"${@:2} --total-time-steps 6000000 --novelty-step 3000000 --env-configs-file simple_to_lava_crossing"}" 
    eval "$command"
done