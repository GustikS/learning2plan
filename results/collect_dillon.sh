#!/bin/bash

# collect csv results from Dillon's cluster

# path to code
PATH_TO_CLUSTER_LAAS_CODE=/pfcalcul/work/dchen/code/cvut-colab
PATH_TO_GADI_CODE=/scratch/cd85/dc6693/cvut-colab

while true; do
    # run cluster-laas to_csv.py
    ssh cluster-laas "cd $PATH_TO_CLUSTER_LAAS_CODE && /pfcalcul/work/dchen/containers/stats.sif ./slurm/to_csv.py"

    # collect cluster-laas csv
    scp cluster-laas:$PATH_TO_CLUSTER_LAAS_CODE/slurm/baseline_results.csv results/baseline_results.csv
    scp cluster-laas:$PATH_TO_CLUSTER_LAAS_CODE/slurm/scorpion_results.csv results/scorpion_results.csv

    # collect gadi logs
    rsync -av --progress gadi:$PATH_TO_GADI_CODE/pbs/__experiments/test_logs results/

    # wait a minute
    sleep 60
done
