#!/bin/bash

# collect csv results from Dillon's cluster

# path to cluster code
PATH_TO_CODE=/pfcalcul/work/dchen/code/cvut-colab

while true; do
    # run cluster's to_csv.py
    ssh cluster-laas "cd $PATH_TO_CODE && /pfcalcul/work/dchen/containers/stats.sif ./slurm/to_csv.py"

    # collect csv
    scp cluster-laas:$PATH_TO_CODE/slurm/baseline_results.csv results/baseline_results.csv
    scp cluster-laas:$PATH_TO_CODE/slurm/scorpion_results.csv results/scorpion_results.csv

    # wait a minute
    sleep 60
done
