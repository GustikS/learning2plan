#!/bin/bash

# collect csv results from Dillon's cluster

# path to cluster
PATH_TO_CODE=cluster-laas:/pfcalcul/work/dchen/code/cvut-colab

# TODO run cluster's to_csv.py

# collect csv
scp $PATH_TO_CODE/slurm/baseline_results.csv results/baseline_results.csv
