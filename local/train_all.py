#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from itertools import product

## paths
# make everything relative to where this script is located
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# assume you have built the container and put it in root directory
CONTAINER = f"{CUR_DIR}/../lrnn_planning.sif"  
assert os.path.exists(CONTAINER), CONTAINER
EXPERIMENTS_DIR = f"{CUR_DIR}/__experiments"
LOG_DIR = f"{EXPERIMENTS_DIR}/train_logs"
SAVE_DIR = f"{EXPERIMENTS_DIR}/models"
LOCK_DIR = f"{CUR_DIR}/.lock_train"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)

PARAMETER_FILE = f"{CUR_DIR}/../parameters.json"
assert os.path.exists(PARAMETER_FILE), PARAMETER_FILE
with open(PARAMETER_FILE, "r") as f:
    parameters = json.load(f)
DIMENSIONS = parameters["dimensions"]
LAYERS = parameters["layers"]
REPEATS = parameters["repeats"]
POLICY_SAMPLE = parameters["policy_sample"]

DOMAINS = [
    "blocksworld", 
    "ferry", 
    "satellite", 
    "rover",
]

CONFIGS = reversed(sorted(product(DOMAINS, LAYERS, DIMENSIONS, REPEATS)))


""" Main loop """
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    submitted = 0
    skipped = 0
    to_go = 0

    for domain, layer, dim, repeat in CONFIGS:
        description = f"{domain}_{layer}_{dim}_{repeat}"

        log_file = f"{LOG_DIR}/{description}.log"
        save_file = f"{SAVE_DIR}/{description}.model"

        if (os.path.exists(save_file)) and not args.force:
            skipped += 1
            # print(log_file)
            continue

        cmd = f"apptainer run {CONTAINER} python3 run.py -d {domain} --embedding {dim} --layers {layer} -s {repeat} --epochs 100 --save_file {save_file}"
        os.system("date")
        print(cmd)
        print(log_file)
        _ = os.popen(f"cd ..; {cmd} > {log_file}").read()
        # cmd = cmd.split()
        # with open(log_file, "w") as f:
        #     subprocess.run(cmd, cwd=f"{CUR_DIR}/..", stdout=f, stderr=f)
        #     f.write("\n")
        #     f.flush()
        #     os.fsync(f.fileno())

    print("Submitted:", submitted)
    print("Skipped:", skipped)
    print("To go:", to_go)


if __name__ == "__main__":
    main()
