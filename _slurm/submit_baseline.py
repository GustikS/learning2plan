#!/usr/bin/env python3
import argparse
import os
import subprocess
from itertools import product

DOMAINS = [
    # "blocksworld", 
    # "ferry", 
    "satellite", 
    # "transport",
]
# PROBLEMS = [f"{x}_{y:02d}" for y in range(1, 31) for x in [0, 1, 2]]
PROBLEMS = [f"{x}_{y:02d}" for y in range(1, 31) for x in [0, 1]]
REPEATS = list(range(10))

## paths
# assume we are in slurm/ directory
CUR_DIR = os.getcwd()  
# assume you have built the container and put it in root directory
CONTAINER = f"{CUR_DIR}/../lrnn_planning.sif"  
assert os.path.exists(CONTAINER), CONTAINER
EXPERIMENTS_DIR = f"{CUR_DIR}/__experiments"
LOG_DIR = f"{EXPERIMENTS_DIR}/baseline_logs"
os.makedirs(LOG_DIR, exist_ok=True)
JOB_SCRIPT = f"{CUR_DIR}/slurm_job.sh"
assert os.path.exists(JOB_SCRIPT), JOB_SCRIPT

""" Main loop """
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-w", "--which", action="store_true", help="see which jobs to go")
    args = parser.parse_args()

    skipped_from_log = 0
    to_go = 0

    submitted = 0

    for domain, problem, repeat in product(DOMAINS, PROBLEMS, REPEATS):
        description = f"{domain}_{problem}_r{repeat}"

        log_file = f"{LOG_DIR}/{description}.log"

        if os.path.exists(log_file) and not args.force:
            skipped_from_log += 1
            continue
        
        cmd = f"apptainer run {CONTAINER} python3 run.py -d {domain} -p {problem} -s {repeat} -c sample -b 10000"

        slurm_vars = ','.join([
            f"CMD={cmd}",
        ])

        job_cmd = [
            "sbatch",
            f"--job-name=BA{description}",
            f"--output={log_file}",
            f"--export={slurm_vars}",
            JOB_SCRIPT,
        ]
        p = subprocess.Popen(job_cmd)
        p.wait()

        print(f" log: {log_file}")
        submitted += 1

    print(f"{submitted=}")
    print(f"{skipped_from_log=}")
    print(f"{to_go=}")


if __name__ == "__main__":
    main()
