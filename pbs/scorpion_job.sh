#!/bin/bash
 
#PBS -P cd85
#PBS -q normal
#PBS -l wd
#PBS -M dongbang4204@gmail.com

module load singularity

set -x

cd /scratch/cd85/dc6693/cvut-colab

tmp_dir=trash_$DESC
mkdir -p $tmp_dir
cp $DOMAIN_FILE $tmp_dir/domain.pddl
cp $PROBLEM_FILE $tmp_dir/problem.pddl
cd $tmp_dir

/scratch/cd85/dc6693/cvut-colab/scorpion.sif \
  --transform-task preprocess-h2 --alias scorpion \
  domain.pddl problem.pddl
