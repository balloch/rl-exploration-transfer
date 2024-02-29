#!/usr/bin/bash
methods=$(echo $1 | tr "," "\n")
for var in $methods
do
    command=$(cat scripts/run_all_tuned_experiments.sh | grep "$var")
    command="${command//\$@/"${@:2} --total-time-steps 7000000 --novelty-step 5000000 --env-configs-file lava_maze_hurt_to_safe"}" 
    eval "$command"
done