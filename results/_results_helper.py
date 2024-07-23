import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

DOMAINS = [
    "blocksworld",
    "ferry",
    "miconic",
    # "rovers",
    "satellite",
    "transport",
]
easy_problems = set(f"0_{i:02d}" for i in range(1, 31))
medium_problems = set(f"1_{i:02d}" for i in range(1, 31))
hard_problems = set(f"2_{i:02d}" for i in range(1, 31))
SKIP_HARD = True
if not SKIP_HARD:
    print("we skip hard problems because they take too long and we don't have time to optimise")
    SKIP_HARD = True

os.makedirs("plots/", exist_ok=True)

this_file_dir = pathlib.Path(__file__).parent.resolve()
print(f"{this_file_dir=}")

baseline_logs_dir = this_file_dir / "baseline_logs"
test_logs_dir = this_file_dir / "test_logs"
raw_logs_exist = os.path.exists(test_logs_dir) and os.path.isdir(test_logs_dir)
if not raw_logs_exist:
    print("No raw logs found. Reading data from csv files...")
else:
    print("Raw logs found. Rewriting data to csv files...")

if raw_logs_exist:
    """ LRNN logs """
    # read test logs and write to csv
    # f"{domain}_{layer}_{dim}_{choice}_{pro_blem}_{repeat}"
    columns = ["domain", "layer", "dim", "choice", "problem", "repeat", "plan_length", "plan_found", "time"]
    data = {k: [] for k in columns}
    for f in sorted(os.listdir("test_logs")):
        if not f.endswith(".log"):
            continue
        toks = f[:-4].split("_")
        domain = toks[0]
        layer = int(toks[1])
        dim = int(toks[2])
        choice = toks[3]
        problem = toks[4] + "_" + toks[5]
        repeat = int(toks[6])
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
        data["choice"].append(choice)
        data["problem"].append(problem)
        data["repeat"].append(repeat)
        data["plan_length"].append(plan_length)
        data["plan_found"].append(solved)
        data["time"].append(time)
    df = pd.DataFrame(data)
    df.to_csv("lrnn_results.csv")

    """ baseline logs """
    log_dir = baseline_logs_dir
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
    df.to_csv(this_file_dir / "baseline_results.csv", index=False)


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
        aggr["choice"] = "first"
    group_by = ["domain", "problem"]
    if lrnn:
        group_by += ["layer", "dim", "choice"]
    df = df.groupby(group_by).agg(aggr)
    df.columns = df.columns.map("_".join)
    df["all_solved"] = df.plan_found_mean == 1
    df.reset_index(inplace=True)
    return df

""" Other logs """
groups = ["baseline", "scorpion", "lrnn"]
datas = {}
for group in groups:
    data = pd.read_csv(f"{group}_results.csv")
    is_lrnn = group == "lrnn"
    data = group_repeats(data, lrnn=is_lrnn)
    data["solver"] = group if group != "scorpion" else "optimal"
    if is_lrnn:
        data["solver"] = "lrnn" + "_L" + data["layer_first"].astype(str) + "_D" + data["dim_first"].astype(str) + "_" + data["choice"]
        data["type"] = "lrnn"
    else:
        data["type"] = "bounds"
    datas[group] = data
# display(datas["lrnn"])

failure_logs = {domain: [] for domain in DOMAINS}
all_data = pd.concat([data for data in datas.values()])
all_data["difficulty"] = all_data["problem"].apply(lambda x: x.split("_")[0])
if SKIP_HARD:
    print("skipping hard problems because they take too long")
    all_data = all_data[~all_data["problem"].isin(hard_problems)]


def plot_domains(metric, log_y=False, include_sample=False, dimensions=None):
    if dimensions is None:
        dimensions = []
    dimensions = set(dimensions)

    solvers = all_data["solver"].unique()

    ignore_models = set()
    if not include_sample:
        # get set of solvers from all_data
        for solver in solvers:
            if "sample" in solver:
                ignore_models.add(solver)
    
    for solver in solvers:
        if not solver.startswith("lrnn"):
            continue
        toks = solver.split("_")
        dimension = int(toks[2][1:])
        if dimension not in dimensions:
            ignore_models.add(solver)

    plot_dir = f"plots/{metric}"
    os.makedirs(plot_dir, exist_ok=True)

    for domain in DOMAINS:
        print(domain)
        data = all_data[all_data["domain"] == domain]
        data = data[~data["solver"].isin(ignore_models)]
        metric_mean = f"{metric}_mean"
        metric_std = f"{metric}_std"
        fig = px.line(data, x="problem", y=metric_mean, error_y=metric_std, color="solver",line_dash="type", log_y=log_y, facet_col="difficulty")
        # fig = px.scatter(data, x="problem", y=metric_mean, error_y=metric_std, color="solver", log_y=log_y)
        fig.update_xaxes(categoryorder='array')
        fig.update_yaxes(matches=None)
        fig.update_xaxes(matches=None)
        fig.for_each_yaxis(lambda y: y.update(showticklabels=True,matches=None))
        fig.for_each_xaxis(lambda x: x.update(showticklabels=True,matches=None))
        fig.write_html(plot_dir + "/" + domain + ".html")
        fig.show()


def plot_difference(absolute=True, include_sample=False, layers=None, dimensions=None):
    if dimensions is None:
        dimensions = []
    dimensions = set(dimensions)
    if layers is None:
        layers = []
    layers = set(layers)

    solvers = all_data["solver"].unique()

    ignore_models = set()
    if not include_sample:
        # get set of solvers from all_data
        for solver in solvers:
            if "sample" in solver:
                ignore_models.add(solver)
    
    for solver in solvers:
        if not solver.startswith("lrnn"):
            continue
        toks = solver.split("_")
        dimension = int(toks[2][1:])
        if dimension not in dimensions:
            ignore_models.add(solver)
        layer = int(toks[1][1:])
        if layer not in layers:
            ignore_models.add(solver)

    plot_dir = f"plots/difference"
    if absolute:
        plot_dir += "_absolute"
    else:
        plot_dir += "_relative"
    os.makedirs(plot_dir, exist_ok=True)

    for domain in DOMAINS:
        print(domain)
        data = all_data[all_data["domain"] == domain]
        data = data[~data["solver"].isin(ignore_models)]

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

        fig = px.line(data, x="problem", y=y, color="solver",line_dash="type", facet_col="difficulty")
        fig.update_xaxes(categoryorder='array')
        fig.update_yaxes(matches=None)
        fig.update_xaxes(matches=None)
        fig.for_each_yaxis(lambda y: y.update(showticklabels=True,matches=None))
        fig.for_each_xaxis(lambda x: x.update(showticklabels=True,matches=None))
        fig.write_html(plot_dir + "/" + domain + ".html")
        fig.show()
