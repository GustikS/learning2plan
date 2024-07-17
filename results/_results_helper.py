import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

DOMAINS = [
    "blocksworld",
    "ferry",
    "miconic",
    # "rovers",
    "satellite",
    "transport",
]

# read test logs and write to csv
# f"{domain}_{layer}_{dim}_{pro_blem}_{repeat}"
columns = ["domain", "layer", "dim", "problem", "repeat", "plan_length", "plan_found", "time"]
data = {k: [] for k in columns}
for f in sorted(os.listdir("test_logs")):
    if not f.endswith(".log"):
        continue
    toks = f[:-4].split("_")
    domain = toks[0]
    layer = int(toks[1])
    dim = int(toks[2])
    problem = toks[3] + "_" + toks[4]
    repeat = int(toks[5])
    with open(f"test_logs/{f}") as f:
        content = f.read()
        solved = "Plan generated!" in content
        if solved:
            plan_length = int(content.split("plan_length=")[1].split("\n")[0])
            time = float(content.split("total_time=")[1].split("\n")[0])
        else:
            plan_length = None
            time = None
    data["domain"].append(domain)
    data["layer"].append(layer)
    data["dim"].append(dim)
    data["problem"].append(problem)
    data["repeat"].append(repeat)
    data["plan_length"].append(plan_length)
    data["plan_found"].append(solved)
    data["time"].append(time)

df = pd.DataFrame(data)
df.to_csv("lrnn_results.csv")


def group_repeats(df, others_to_keep=None, lrnn=False):
    if others_to_keep is None:
        others_to_keep = []
    aggr = {
        "time": ["mean", "std"],
        "plan_length": ["mean", "std"],
        "plan_found": "mean",
    }
    if lrnn:
        aggr["layer"] = "first"
        aggr["dim"] = "first"
    group_by = ["domain", "problem"]
    if lrnn:
        group_by += ["layer", "dim"]
    df = df.groupby(group_by).agg(aggr)
    df.columns = df.columns.map("_".join)
    df["all_solved"] = df.plan_found_mean == 1
    df.reset_index(inplace=True)
    return df


groups = ["baseline", "scorpion", "lrnn"]
datas = {}
for group in groups:
    data = pd.read_csv(f"{group}_results.csv")
    is_lrnn = group == "lrnn"
    data = group_repeats(data, lrnn=is_lrnn)
    data["solver"] = group
    if is_lrnn:
        data["solver"] = "lrnn" + "_L" + data["layer_first"].astype(str) + "_D" + data["dim_first"].astype(str)
    datas[group] = data
# display(datas["lrnn"])

failure_logs = {domain: [] for domain in DOMAINS}
all_data = pd.concat([data for data in datas.values()])
easy_problems = set(f"0_{i:02d}" for i in range(1, 31))


def plot_domains(metric, log_y=False, easy_only=False):
    for domain in DOMAINS:
        print(domain)
        data = all_data[all_data["domain"] == domain]
        if easy_only:
            data = data[data["problem"].isin(easy_problems)]
        # failures = data[data.plan_found == False]
        # max_bound_failures = data[data.max_bound == False]
        # failures_not_due_to_max_bound = data[(data.plan_found == False) & (data.max_bound == False)]
        # n_failures = len(failures)
        # n_max_bound_failures = len(max_bound_failures)
        # n_failures_not_due_to_max_bound = len(failures_not_due_to_max_bound)
        # print(f"{n_failures=}")
        # for row in failures.itertuples():
        #     # print(row)
        #     failure_logs[domain].append(f"/pfcalcul/work/dchen/code/cvut-colab/slurm/__experiments/baseline_logs/{row.domain}_{row.problem}_r{row.repeat}.log")
        metric_mean = f"{metric}_mean"
        metric_std = f"{metric}_std"
        fig = px.line(data, x="problem", y=metric_mean, error_y=metric_std, color="solver", log_y=log_y)
        # fig = px.scatter(data, x="problem", y=metric_mean, error_y=metric_std, color="solver", log_y=log_y)
        fig.show()


def plot_difference(absolute=True):
    for domain in DOMAINS:
        print(domain)
        data = all_data[all_data["domain"] == domain]

        data["improvement"] = data.apply(
            lambda row: -row["plan_length_mean"]
            + data[
                (data["domain"] == row["domain"])
                & (data["problem"] == row["problem"])
                & (data["solver"] == "baseline")
            ]["plan_length_mean"].values[0],
            axis=1,
        )
        data["improvement (%)"] = data.apply(
            lambda row: 100 * (
                -row["plan_length_mean"]
                + data[
                    (data["domain"] == row["domain"])
                    & (data["problem"] == row["problem"])
                    & (data["solver"] == "baseline")
                ]["plan_length_mean"].values[0]
            )
            / data[
                (data["domain"] == row["domain"])
                & (data["problem"] == row["problem"])
                & (data["solver"] == "baseline")
            ]["plan_length_mean"].values[0],
            axis=1,
        )

        if absolute:
            y = "improvement"
        else:
            y = "improvement (%)"

        fig = px.line(data, x="problem", y=y, color="solver")
        fig.show()
