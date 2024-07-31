import json
import os
import pathlib
from itertools import product

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

LRNN_REPEATS = {
    0,
    1,
    2,
    3,
    4,
}
TAKE_BEST = 0

DOMAINS = [
    "blocksworld",
    "ferry",
    "satellite",
    "rover",
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
planner_logs_dir = this_file_dir / "planner_logs"
raw_logs_exist = os.path.exists(test_logs_dir) and os.path.exists(baseline_logs_dir) and os.path.exists(planner_logs_dir)
if not raw_logs_exist:
    print("No raw logs found. Reading data from csv files...")
else:
    print("Raw logs found. Rewriting data to csv files...")

if raw_logs_exist:
    """ LRNN logs """
    # read test logs and write to csv
    # f"{domain}_{layer}_{dim}_{choice}_{pro_blem}_{repeat}"
    columns = ["domain", "layer", "dim", "choice", "problem", "repeat", "cycles", "plan_length", "plan_found", "time"]
    data = {k: [] for k in columns}
    for file in sorted(os.listdir("test_logs")):
        if not file.endswith(".log"):
            continue
        try:
            toks = file[:-4].split("_")
            domain = toks[0]
            layer = int(toks[1])
            dim = int(toks[2])
            choice = toks[3]
            problem = toks[4] + "_" + toks[5]
            repeat = int(toks[6])
            with open(f"test_logs/{file}") as f:
                content = f.read()
                solved = "Plan generated!" in content
                if solved:
                    plan_length = int(content.split("plan_length=")[1].split("\n")[0])
                    time = float(content.split("total_time=")[1].split("\n")[0])
                    cycles = int(content.split("cycles_detected=")[-1].split("\n")[0])
                else:
                    plan_length = None
                    time = None
                    cycles = None
        except Exception as e:
            print(f"Error in {file}: {e}")
            continue
        if repeat not in LRNN_REPEATS:
            continue
        data["domain"].append(domain)
        data["layer"].append(layer)
        data["dim"].append(dim)
        data["choice"].append(choice)
        data["problem"].append(problem)
        data["repeat"].append(repeat)
        data["cycles"].append(cycles)
        data["plan_length"].append(plan_length)
        data["plan_found"].append(solved)
        data["time"].append(time)
    df = pd.DataFrame(data)
    df.to_csv("lrnn_results.csv")

    """ scorpion and lama logs"""
    bw_opt_plan_lengths = [10,8,20,24,24,26,32,32,36,38,48,40,42,44,46,54,60,50,54,56,56,72,58,72,82,74,66,82,76,88,106,104,112,144,142,140,158,164,180,186,202,234,240,228,232,246,272,284,286,308,310,314,354,338,348,388,380,368,386,464]
    log_dir = planner_logs_dir
    keys = ["domain", "problem", "time", "plan_length", "plan_found"]
    data = {planner: {k: [] for k in keys} for planner in ["lama", "scorpion"]}
    # pbar = tqdm(sorted(os.listdir(log_dir)))
    # for f in pbar:
    for f in sorted(os.listdir(log_dir)):
        log_f = log_dir / f
        # pbar.set_description(str(log_f))
        toks = f.replace(".log", "").split("_")
        domain = toks[0]
        problem = toks[1] + "_" + toks[2]
        planner = toks[3]
        if domain=="blocksworld" and planner=="scorpion":  # used slaney thiebaux planner
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
        data[planner]["domain"].append(domain)
        data[planner]["problem"].append(problem)
        data[planner]["time"].append(time)
        data[planner]["plan_length"].append(plan_length)
        data[planner]["plan_found"].append(plan_found)
    
    for planner in ["lama", "scorpion"]:
        df = pd.DataFrame(data[planner])
        df.to_csv(f"{planner}_results.csv")

    """ baseline logs """
    log_dir = baseline_logs_dir
    keys = ["domain", "problem", "repeat", "time", "cycles", "plan_length", "plan_found", "max_bound"]
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
                cycles = np.nan
                max_bound = "Terminating with failure after" in content
            else:
                time = float((content.split("total_time=")[1]).split("\n")[0])
                plan_length = float((content.split("plan_length=")[1]).split("\n")[0])
                cycles = int(content.split("cycles_detected=")[-1].split("\n")[0])
                max_bound = False
        data["domain"].append(domain)
        data["problem"].append(problem)
        data["repeat"].append(repeat)
        data["time"].append(time)
        data["cycles"].append(cycles)
        data["plan_length"].append(plan_length)
        data["plan_found"].append(plan_found)
        data["max_bound"].append(max_bound)
    
    df = pd.DataFrame(data)
    df.to_csv(this_file_dir / "baseline_results.csv", index=False)


def group_repeats(df, others_to_keep=None, lrnn=False):
    if others_to_keep is None:
        others_to_keep = []
    if "cycles" not in df.columns:
        df["cycles"] = 0
    if TAKE_BEST:
        aggr = {
            "time": ["min"],
            "plan_length": ["min"],
            "cycles": ["min"],
            "plan_found": "mean",
        }
    else:
        aggr = {
            "time": ["mean", "std"],
            "plan_length": ["mean", "std"],
            "cycles": ["mean", "std"],
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


def visualise_cylces():
    data = pd.read_csv(f"baseline_results.csv")
    print("Ideally, we want 0 cycles")
    fig = px.line(data, x="problem", y="cycles", color="domain", log_y=True)
    fig.show()


""" Other logs """
groups = ["baseline", "scorpion", "lama", "lrnn"]
datas = {}
for group in groups:
    data = pd.read_csv(f"{group}_results.csv")
    is_lrnn = group == "lrnn"
    data = group_repeats(data, lrnn=is_lrnn)
    data["solver"] = group if group != "scorpion" else "optimal"
    if is_lrnn:
        data["solver"] = "L" + data["layer_first"].astype(str) + "_D" + data["dim_first"].astype(str) + "_" + data["choice"]
        data["type"] = "lrnn_" + data["choice"]
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


def get_ignore_models(choices=None, layers=None, dimensions=None):
    if choices is None:
        choices = []
    choices = set(choices)

    if dimensions is None:
        dimensions = []
    dimensions = set(dimensions)

    if layers is None:
        layers = []
    layers = set(layers)

    solvers = all_data["solver"].unique()

    ignore_models = set()
    for solver in solvers:
        if not solver.startswith("L"):
            continue
        toks = solver.split("_")
        choice = toks[2]
        if choice not in choices:
            ignore_models.add(solver)

        dimension = int(toks[1][1:])
        if dimension not in dimensions:
            ignore_models.add(solver)

        layer = int(toks[0][1:])
        if layer not in layers:
            ignore_models.add(solver)
    
    return ignore_models


def get_improvement(df):
    combination = "plan_length_mean" if not TAKE_BEST else "plan_length_min"
    df["improvement"] = df.apply(
        lambda row: -row[combination]
        + df[
            (df["domain"] == row["domain"])
            & (df["problem"] == row["problem"])
            & (df["solver"] == "baseline")
        ][combination].values[0],
        axis=1,
    )
    df["improvement (%)"] = df.apply(
        lambda row: 100 * (
            -row[combination]
            + df[
                (df["domain"] == row["domain"])
                & (df["problem"] == row["problem"])
                & (df["solver"] == "baseline")
            ][combination].values[0]
        )
        / df[
            (df["domain"] == row["domain"])
            & (df["problem"] == row["problem"])
            & (df["solver"] == "baseline")
        ][combination].values[0],
        axis=1,
    )
    return df


def quantify_performance(choices=None, layers=None, dimensions=None):
    assert len(choices) == 1
    ignore_models = get_ignore_models(choices, layers, dimensions)
    for domain in DOMAINS:
        data = all_data[all_data["domain"] == domain]
        data = data[~data["solver"].isin(ignore_models)]
        print(domain)
        df = get_improvement(data)
        df = df[~df["type"].isin(["bounds"])]

        # print(df)
        df["config"] = "" + df["layer_first"].astype(int).astype(str) + "_" + df["dim_first"].astype(int).astype(str)
        group_by = ["config", "difficulty"]
        df = df.groupby(group_by).agg({"improvement (%)": ["mean", "std"]})
        df.columns = df.columns.map(" ".join)
        df.reset_index(inplace=True)
        fig = px.scatter(df, x="config", y="improvement (%) mean", error_y="improvement (%) std", facet_col="difficulty")
        fig.update_yaxes(range=[-20, 20])
        fig.show()

        # easy = data[data["difficulty"] == "0"]
        # medium = data[data["difficulty"] == "1"]
        # easy_mean = data["improvement"].mean()
        # easy_std = data["improvement"].std()
        # medium_mean = medium["improvement"].mean()
        # medium_std = medium["improvement"].std()
        # print(f"{easy_mean=}")
        # print(f"{easy_std=}")
        # print(f"{medium_mean=}")
        # print(f"{medium_std=}")


def plot_domains(metric, log_y=False, choices=None, layers=None, dimensions=None):
    ignore_models = get_ignore_models(choices, layers, dimensions)

    plot_dir = f"plots/{metric}"
    os.makedirs(plot_dir, exist_ok=True)

    for domain in DOMAINS:
        print(domain)
        data = all_data[all_data["domain"] == domain]
        data = data[~data["solver"].isin(ignore_models)]
        metric_mean = f"{metric}_mean"
        metric_std = f"{metric}_std"
        metric_best = f"{metric}_min"
        if TAKE_BEST:
            fig = px.line(data, x="problem", y=metric_best, color="solver",line_dash="type", log_y=log_y, facet_col="difficulty")
        else:
            fig = px.line(data, x="problem", y=metric_mean, error_y=metric_std, color="solver", line_dash="type", log_y=log_y, facet_col="difficulty", category_orders={"problem": sorted(easy_problems | medium_problems)})
        # fig.update_xaxes(categoryorder='array')
        fig.update_yaxes(matches=None)
        fig.update_xaxes(matches=None)
        fig.for_each_yaxis(lambda y: y.update(showticklabels=True,matches=None))
        fig.for_each_xaxis(lambda x: x.update(showticklabels=True,matches=None))
        fig.write_html(plot_dir + "/" + domain + ".html")
        fig.show()


def plot_difference(absolute=True, choices=None, layers=None, dimensions=None):
    ignore_models = get_ignore_models(choices, layers, dimensions)

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
        data = get_improvement(data)

        if absolute:
            y = "improvement"
        else:
            y = "improvement (%)"

        fig = px.line(data, x="problem", y=y, color="solver",line_dash="type", facet_col="difficulty")
        fig.update_xaxes(categoryorder='array')
        fig.update_yaxes(matches=None)
        fig.update_xaxes(matches=None)
        fig.update_yaxes(range=[-100, 100])
        fig.for_each_yaxis(lambda y: y.update(showticklabels=True,matches=None))
        fig.for_each_xaxis(lambda x: x.update(showticklabels=True,matches=None))
        fig.write_html(plot_dir + "/" + domain + ".html")
        fig.show()


##################### Train stats
# make everything relative to where this script is located
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMETER_FILE = f"{CUR_DIR}/../parameters.json"
assert os.path.exists(PARAMETER_FILE), PARAMETER_FILE
with open(PARAMETER_FILE, "r") as f:
    parameters = json.load(f)
DIMENSIONS = parameters["dimensions"]
LAYERS = parameters["layers"]
LRNN_REPEATS = parameters["repeats"]
POLICY_SAMPLE = parameters["policy_sample"]

TRAIN_LOG_DIR = f"{this_file_dir}/train_logs"

train_csv_file = f"{this_file_dir}/train_results.csv"

train_data = ["domain", "layers", "dimension", "config", "repeat", "loss", "f1", "epoch", "time"]
train_data = {k: [] for k in train_data}

if os.path.exists(TRAIN_LOG_DIR):
    for domain, layer, dimension, repeat in product(DOMAINS, LAYERS, DIMENSIONS, LRNN_REPEATS):
        log_file = f"{TRAIN_LOG_DIR}/{domain}_{layer}_{dimension}_{repeat}.log"
        if not os.path.exists(log_file):
            continue
        try:
            with open(log_file, "r") as f:
                content = f.read()
                epoch = content.split("Best model at epoch=")[1].split()[0]
                t = content.split("Finished training the LRNN in")[1].split("s\n")[0]
                content = content.split(f"epoch={epoch}, ")[1].split("\n")[0]
                toks = content.split(", ")
                loss = float(toks[1].replace("loss=np.float64", "").replace("(", "").replace(")", ""))
                f1 = float(toks[3].replace("f1=np.float64", "").replace("(", "").replace(")", ""))
        except:
            print(f"Training failure in {log_file}")
            continue
        train_data["domain"].append(domain)
        train_data["layers"].append(layer)
        train_data["dimension"].append(dimension)
        train_data["repeat"].append(repeat)
        train_data["config"].append(f"{layer}_{dimension}")
        train_data["loss"].append(loss)
        train_data["f1"].append(f1)
        train_data["epoch"].append(epoch)
        train_data["time"].append(t)
        # print(domain, layer, dimension, repeat, loss, f1, epoch, t)

    train_df = pd.DataFrame(train_data)
    train_df.to_csv(train_csv_file, index=False)
else:
    print("Using existing train csv")
    train_df = pd.read_csv(train_csv_file)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualise_train(layers, dimensions):
    layers = set(layers)
    dimensions = set(dimensions)
    for domain in DOMAINS:
        print(domain)
        # display(train_df)
        domain_df = train_df[train_df["domain"] == domain]
        domain_df = domain_df[domain_df["layers"].isin(layers)]
        domain_df = domain_df[domain_df["dimension"].isin(dimensions)]
        domain_df["time"] = domain_df["time"].astype(float)
        fig = make_subplots(rows=1, cols=3)

        fig.add_trace(
            go.Scatter(x=domain_df["config"], y=domain_df["loss"],mode='markers',name="loss",hoverinfo=["all"]),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=domain_df["config"], y=domain_df["f1"],mode='markers',name="f1",hoverinfo=["all"]),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=domain_df["config"], y=domain_df["time"],mode='markers',name="time",hoverinfo=["all"]),
            row=1, col=3
        )

        # fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
        fig.show()
