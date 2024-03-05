#!/bin/bash
methods=$(echo $1 | tr "," "\n")
for var in $methods
do
    command=$(cat scripts/run_all_tuned_experiments.sh | grep "$var")
    command="${command//\$@/"${@:2} --total-time-steps 10500000 --novelty-step 500000 --env-configs-file lava_maze_safe_to_hurt"}" 
    eval "$command"
done