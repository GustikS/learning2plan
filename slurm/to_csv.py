#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pathlib
import os
from tqdm import tqdm

def main():
    file_dir = pathlib.Path(__file__).parent.resolve()
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
    
    df = pd.DataFrame(data)
    df.to_csv(file_dir / "baseline_results.csv", index=False)

if __name__ == "__main__":
    main()
