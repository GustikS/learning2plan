#!/bin/bash

# path to code
PATH_TO_GADI_CODE=/scratch/cd85/dc6693/cvut-colab

while true; do
    rsync -av --progress __experiments/models gadi:$PATH_TO_GADI_CODE/pbs/__experiments/

    date

    # wait a minute
    sleep 60
done
