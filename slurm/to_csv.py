#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pathlib
import os
from tqdm import tqdm


file_dir = pathlib.Path(__file__).parent.resolve()

def baselines():
    log_dir = file_dir / "__experiments" / "baseline_logs"
    print(log_dir)
    keys = ["domain", "problem", "repeat", "time", "plan_length", "plan_found", "max_bound"]
    data = {k: [] for k in keys}
    pbar = tqdm(sorted(os.listdir(log_dir)))
    for f in pbar:
        log_f = log_dir / f
        pbar.set_description(str(log_f))
        toks = f.split("_")
        domain = toks[0]
        problem = toks[1] + "_" + toks[2]
        repeat = toks[3][1]
        with open(log_f, "r") as f:
            content = f.read()
            plan_found = "Plan generated!" in content
            if not plan_found:
                time = np.nan
                plan_length = np.nan
                max_bound = "Terminating with failure after" in content
            else:
                time = float((content.split("total_time=")[1]).split("\n")[0])
                plan_length = float((content.split("plan_length=")[1]).split("\n")[0])
                max_bound = False
        data["domain"].append(domain)
        data["problem"].append(problem)
        data["repeat"].append(repeat)
        data["time"].append(time)
        data["plan_length"].append(plan_length)
        data["plan_found"].append(plan_found)
        data["max_bound"].append(max_bound)
    
    df = pd.DataFrame(data)
    df.to_csv(file_dir / "baseline_results.csv", index=False)

def scorpion():
    bw_opt_plan_lengths = [10,8,20,24,24,26,32,32,36,38,48,40,42,44,46,54,60,50,54,56,56,72,58,72,82,74,66,82,76,88,106,104,112,144,142,140,158,164,180,186,202,234,240,228,232,246,272,284,286,308,310,314,354,338,348,388,380,368,386,464]
    log_dir = file_dir / "__experiments" / "scorpion_logs"
    print(log_dir)
    keys = ["domain", "problem", "time", "plan_length", "plan_found"]
    data = {k: [] for k in keys}
    # pbar = tqdm(sorted(os.listdir(log_dir)))
    # for f in pbar:
    for f in sorted(os.listdir(log_dir)):
        log_f = log_dir / f
        # pbar.set_description(str(log_f))
        toks = f.split("_")
        domain = toks[0]
        problem = toks[1] + "_" + toks[2]
        problem = problem.replace(".log", "")
        if domain=="blocksworld":  # used slaney thiebaux planner
            toks = problem.split("_")
            diff = int(toks[0])
            prob = int(toks[1])
            index = diff * 30 + prob - 1
            plan_found = True
            if index >= len(bw_opt_plan_lengths):
                plan_length = np.nan
                time = np.nan
            else:
                plan_length = bw_opt_plan_lengths[index]
                time = 0
        else:
            with open(log_f, "r") as f:
                content = f.read()
                plan_found = "Solution found!" in content
                if not plan_found:
                    time = np.nan
                    plan_length = np.nan
                else:
                    time = float((content.split("Total time: ")[1].replace("s", "")).split("\n")[0])
                    plan_length = float((content.split("Plan length: ")[1]).split()[0])
        data["domain"].append(domain)
        data["problem"].append(problem)
        data["time"].append(time)
        data["plan_length"].append(plan_length)
        data["plan_found"].append(plan_found)
    
    df = pd.DataFrame(data)
    df.to_csv(file_dir / "scorpion_results.csv", index=False)


def main():
    # baselines()
    # scorpion()
    print(f"Doing nothing. Uncomment code to run.")

if __name__ == "__main__":
    main()
