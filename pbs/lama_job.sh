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
  --transform-task preprocess-h2 \
  domain.pddl problem.pddl \
  --search "let(hlm, landmark_sum(lm_reasonable_orders_hps(lm_rhw()), transform=adapt_costs(one)),
    let(hff, ff(transform=adapt_costs(one)),
    lazy(alt([single(hff), single(hff, pref_only=true), single(hlm), single(hlm, pref_only=true),
    type_based([hff, g()])], boost=1000), preferred=[hff, hlm], cost_type=one)))"
