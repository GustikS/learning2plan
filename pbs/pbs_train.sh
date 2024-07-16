#!/bin/bash
 
#PBS -P cd85
#PBS -q normal
#PBS -l wd
#PBS -M dongbang4204@gmail.com

module load singularity

set -x

cd /scratch/cd85/dc6693/cvut-colab

/scratch/cd85/dc6693/cvut-colab/lrnn_planning.sif $CMD
