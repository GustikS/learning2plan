#!/bin/bash

# collect csv results from Dillon's cluster

set -e

set -x

# path to code
PATH_TO_CLUSTER_LAAS_CODE=/pfcalcul/work/dchen/code/cvut-colab
PATH_TO_GADI_CODE=/scratch/cd85/dc6693/cvut-colab

while true; do
    mkdir -p baseline_logs

    # BK logs from gadi
    rsync -av --progress gadi:$PATH_TO_GADI_CODE/pbs/__experiments/baseline_logs results

    # planner logs from gadi
    rsync -av --progress gadi:$PATH_TO_GADI_CODE/pbs/__experiments/planner_logs results

    # train logs local
    cp -r local/__experiments/train_logs results/

    # train logs from cluster-laas
    rsync -av --progress cluster-laas:$PATH_TO_CLUSTER_LAAS_CODE/slurm/__experiments/train_logs results/

    # test logs from cluster-laas
    rsync -av --progress cluster-laas:$PATH_TO_CLUSTER_LAAS_CODE/slurm/__experiments/test_logs results/

    # # collect test logs from gadi
    # rsync -av --progress gadi:/scratch/cd85/dc6693/cvut-colab/pbs/__experiments/test_logs results/

    # date
    date

    # wait a minute
    sleep 60
done
