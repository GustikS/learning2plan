import os

for i in [0, 1, 2]:
    for j in range(1, 31):
        out_file = f"testing/p{i}_{j:02d}.pddl"
        n_blocks = 10 + j + i * 30
        os.system(f"python3 blocks-world-generator-and-planner/bbwstates_src/to_pddl.py -n {n_blocks} -o {out_file}")
