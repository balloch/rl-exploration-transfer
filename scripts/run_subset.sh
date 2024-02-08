#!/usr/bin/bash
methods=$(echo $1 | tr "," "\n")
for var in $methods
do
    command=$(cat scripts/batch_run_ir_experiments.sh | grep "$var")
    command=${command%\$@}
    eval "$command ${@:2}"
done