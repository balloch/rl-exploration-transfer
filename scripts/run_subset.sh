#!/usr/bin/bash
methods=$(echo $1 | tr "," "\n")
for var in $methods
do
    command=$(cat scripts/run_all_experiments.sh | grep "$var")
    command=${command%\$@}
    echo "$command ${@:2}"
done