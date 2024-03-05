#!/bin/bash
methods=$(echo $1 | tr "," "\n")
for var in $methods
do
    command=$(cat scripts/run_all_sweeps.sh | grep "$var")
    command="${command//\$@/"${@:2}"}" 
    eval "$command"
done