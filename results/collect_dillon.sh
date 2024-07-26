#!/bin/bash

# collect csv results from Dillon's cluster

set -x

# path to code
PATH_TO_CLUSTER_LAAS_CODE=/pfcalcul/work/dchen/code/cvut-colab
PATH_TO_GADI_CODE=/scratch/cd85/dc6693/cvut-colab

while true; do
    mkdir -p baseline_logs

    # # run cluster-laas to_csv.py
    # ssh cluster-laas "cd $PATH_TO_CLUSTER_LAAS_CODE && /pfcalcul/work/dchen/containers/stats.sif ./slurm/to_csv.py"

    # # collect cluster-laas csv
    # scp cluster-laas:$PATH_TO_CLUSTER_LAAS_CODE/slurm/baseline_results.csv results/baseline_results.csv
    # scp cluster-laas:$PATH_TO_CLUSTER_LAAS_CODE/slurm/scorpion_results.csv results/scorpion_results.csv

    # get logs directly

    # # baseline and optimal logs from cluster-laas
    # rsync -av --progress cluster-laas:$PATH_TO_CLUSTER_LAAS_CODE/slurm/__experiments/baseline_logs results

    # baseline and optimal logs from gadi
    # rsync -av --progress gadi:$PATH_TO_GADI_CODE/pbs/__experiments/baseline_logs results

    # logs from cluster-laas
    rsync -av --progress cluster-laas:$PATH_TO_CLUSTER_LAAS_CODE/slurm/__experiments/test_logs results/

    # # collect test logs from gadi
    # rsync -av --progress gadi:/scratch/cd85/dc6693/cvut-colab/pbs/__experiments/test_logs results/

    # date
    date

    # wait a minute
    sleep 60
done
