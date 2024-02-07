#!/usr/bin/bash
for var in "$@"
do
    cat scripts/batch_run_ir_experiments.sh | grep "$var"
done