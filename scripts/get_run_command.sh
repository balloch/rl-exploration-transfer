#!/usr/bin/bash
for var in "$@"
do
    cat scripts/run_all_experiments.sh | grep "$var"
done