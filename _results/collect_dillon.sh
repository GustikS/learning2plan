#!/bin/bash

# collect csv results from Dillon's cluster

set -e

set -x

# path to code
PATH_TO_CLUSTER_LAAS_LOGS=/pfcalcul/work/dchen/code/cvut-colab/slurm/__experiments
PATH_TO_GADI_LOGS=/scratch/cd85/dc6693/cvut-colab/pbs/__experiments

while true; do
    mkdir -p baseline_logs

    ### BK logs from gadi
    rsync -av --progress gadi:$PATH_TO_GADI_LOGS/baseline_logs results

    ### planner logs from gadi
    # rsync -av --progress gadi:$PATH_TO_GADI_LOGS/planner_logs results

    ### train logs from cluster-laas
    rsync -av --progress cluster-laas:$PATH_TO_CLUSTER_LAAS_LOGS/train_logs results/

    ### train logs local (higher priority)
    cp -r local/__experiments/train_logs results/

    ### test logs from cluster-laas
    # rsync -av --progress cluster-laas:$PATH_TO_CLUSTER_LAAS_LOGS/test_logs results/

    ### collect test logs from gadi
    rsync -av --progress gadi:$PATH_TO_GADI_LOGS/test_logs results/

    # date
    date

    # wait a minute
    sleep 60
done
