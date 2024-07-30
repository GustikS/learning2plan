#!/bin/bash

# path to code
PATH_TO_CLUSTER_CODE=/pfcalcul/work/dchen/code/cvut-colab

while true; do
    rsync -av --progress __experiments/models cluster-laas:$PATH_TO_CLUSTER_CODE/slurm/__experiments/

    date

    # wait a minute
    sleep 60
done
