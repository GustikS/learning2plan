#!/usr/bin/env python

import argparse
import os
import random
import sys
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Union

import neuralogic
import numpy as np
import pymimir
from neuralogic.logging import Formatter, Level, add_handler
from neuralogic.nn.java import NeuraLogic
from termcolor import colored

from datasets.to_jsons import convert_to_json
from modelling.samples import prepare_training_data
from plan.core import PolicyExecutor
from policy_rules.policy.policy_learning import LearningPolicy
from policy_rules.util import template_settings

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

if not neuralogic.is_initialized():
    jar_path = f"{CUR_DIR}/jar/NeuraLogic.jar"
    # custom momentary backend upgrades
    neuralogic.initialize(
        jar_path=jar_path,
        debug_mode=False,  # for java backend debugging
        max_memory_size=64,  # in GB (increase for full miconic)
    )

from policy_rules.policy.handcraft.handcraft_factory import get_handcraft_policy
from policy_rules.util.printing import print_mat
from policy_rules.util.template_settings import load_stored_model
from policy_rules.util.timer import TimerContextManager


def parse_args():
    # fmt: off

    parser = argparse.ArgumentParser()

    # Driver arguments
    parser.add_argument("-d", "--domain", type=str, default="blocksworld",
                        help="Planning domain.")
    parser.add_argument("-p", "--problem", type=str, default="0_01", 
                        help="Planning task. of the form 'x_yy'")
    parser.add_argument("-b", "--bound", type=int, default=100, 
                        help="Policy execution termination bound")

    parser.add_argument("--train", type=bool, default=False, action=BooleanOptionalAction,
                        help="Train an LRNN. Training is also performed if a save file is specified.")
    parser.add_argument("-load", "--load_file", type=str, default=None, 
                        help="Filename to load the template")
    parser.add_argument("-save", "--save_file", type=str, default=None,
                        help="Filename to save the template")

    # Data arguments
    parser.add_argument("-lim", "--limit", type=int, default=-1,
                        help="Training data samples cutoff limit (good for quicker debugging)")
    parser.add_argument("-sr", "--state_regression", type=bool, default=False, action=BooleanOptionalAction,
                        help="Include state h distance labels for (classic) state regression training")
    parser.add_argument("-ar", "--action_regression", type=bool, default=False, action=BooleanOptionalAction,
                        help="Switch between regression/classification labels for output actions in training")
    parser.add_argument("-dnj", "--do_not_json", type=bool, default=False, action=BooleanOptionalAction,
                        help="Do not convert PDDL data to JSON and use existing JSON data.")
    parser.add_argument("-ca", "--cache", type=bool, default=False, action=BooleanOptionalAction,
                        help="Store or use stored built samples. (NotImplemented] Does not work because java objects are not picklable")

    # Model training arguments
    parser.add_argument("-e", "--embedding", type=int, default=1,
                        help="Embedding dimensionality throughout the model (-1 = off, 1 = scalar)")
    parser.add_argument("-l", "--layers", type=int, default=1,
                        help="Number of model layers (-1 = off, 1 = just embedding, 2+ = message-passing)")
    parser.add_argument("-gnn", "--gnn_type", type=str, default="SAGE", choices=["SAGE", "GIN", "TAG"],
                        help="GNN message passing scheme")
    parser.add_argument("-agg", "--aggregation", default="max", choices=["sum", "mean", "max"],
                        help="Aggregation function for message passing")
    parser.add_argument("-ep", "--epochs", type=int, default=100,
                        help="Number of model training epochs")
    parser.add_argument("-k", "--knowledge", type=bool, default=True, action=BooleanOptionalAction,
                        help="An option to skip the domain knowledge and use just a generic ML model")
    parser.add_argument("-vis", "--visualise", default=None,
                        help="Save visualisation of template to file.")
    parser.add_argument("-s", "--seed", type=int, default=2024, 
                        help="Random seed.")
    parser.add_argument("-c", "--choice", default="best", choices=["sample", "best"],
                        help="Choose the best action or sample from the policy. Has no effect for baseline policy which defaults to uniform sampling.")

    # Debugging options
    parser.add_argument("-v", "--verbosity", type=int, default=0,
                        help="Verbosity level for logging")
    parser.add_argument("-eval_bk", "--eval_bk_policy", type=bool, default=False, action=BooleanOptionalAction,
                        help="Computes percentages from training data for the baseline BK policy")

    opts = parser.parse_args()
    return opts
    # fmt: on


def main():
    opts = parse_args()
    seed = opts.seed
    random.seed(seed)
    neuralogic.manual_seed(seed)
    np.random.seed(seed)
    domain_name = opts.domain
    problem_name = opts.problem
    load_file_name = opts.load_file
    save_file_name = opts.save_file
    to_train = ((not load_file_name) and save_file_name) or opts.train or opts.eval_bk_policy
    samples_limit = opts.limit
    state_regression = opts.state_regression
    action_regression = opts.action_regression
    embed_dim = opts.embedding
    gnn_type = opts.gnn_type
    num_layers = opts.layers
    aggregation = opts.aggregation
    num_epochs = opts.epochs
    include_knowledge = opts.knowledge
    domain_path = f"{CUR_DIR}/datasets/pddl/{domain_name}/domain.pddl"
    test_problem_path = f"{CUR_DIR}/datasets/pddl/{domain_name}/testing/p{problem_name}.pddl"
    training_data_path = f"{CUR_DIR}/datasets/lrnn/{domain_name}/classic/data"
    _DEBUG_LEVEL = opts.verbosity
    assert Path(domain_path).exists(), f"Domain file not found: {domain_path}"

    # determine if we are running the baseline handcrafted policy
    run_baseline = not to_train and not load_file_name
    if run_baseline:
        template_settings.debug_settings()
        print(colored(f"Running uniform sampling of the baseline handcrafted policy for {domain_name}", "green"))
        embed_dim = 1

    if to_train:
        template_settings.train_settings()
        print(colored("Running the training script with the following parameters", "green"))
        print(f"    {domain_name=}")
        print(f"    {embed_dim=}")
        print(f"    {num_layers=}")
        print(f"    {aggregation=}")
        print(f"    {num_epochs=}")
        print(f"    {include_knowledge=}")
        print(f"    {seed=}")
        if save_file_name:
            print(f"    {save_file_name=}")

    if _DEBUG_LEVEL > 5:
        add_handler(sys.stdout, Level.FINE, Formatter.COLOR)
    elif _DEBUG_LEVEL > 4:
        add_handler(sys.stdout, Level.WARNING, Formatter.COLOR)
    else:
        add_handler(sys.stdout, Level.OFF, Formatter.COLOR)

    total_time = 0

    """ 1. handle domain information """
    with TimerContextManager(f"parsing PDDL domain file") as timer:
        print(f"{domain_path=}")
        domain = pymimir.DomainParser(str(domain_path)).parse()
        total_time += timer.get_time()

    # possibly load an initial template from file with the same template_name if found
    if load_file_name:
        print(f"Loading policy from {load_file_name}")
        loaded_model: NeuraLogic = load_stored_model(load_file_name)
    else:
        loaded_model = None

    policy: LearningPolicy = get_handcraft_policy(domain_name)(domain, debug=_DEBUG_LEVEL)

    with TimerContextManager("initialising policy template") as timer:
        # no need to recreate the template with every new state, we can retain it for the whole domain
        policy.init_template(
            loaded_model,
            dim=embed_dim,
            num_layers=num_layers,
            include_knowledge=include_knowledge,
            add_types=True,
            state_regression=state_regression,
            action_regression=action_regression,
            gnn_type=gnn_type,
        )
        if _DEBUG_LEVEL > 0:
            policy._debug_template(serialise_file=f"{domain_name}_debug.template.txt")
        total_time += timer.get_time()

    # save visualisation of the template if requested
    save_drawing = opts.visualise
    if save_drawing is not None:
        policy.model.draw(filename=save_drawing)
        print(colored(f"Saved template visualisation to {save_drawing}", "green"))

    # training should be performed if there are training data AND the policy has learnable parameters/model
    if to_train and hasattr(policy, "model"):
        target_subdir = f"data_{opts.layers}_{opts.embedding}_{opts.seed}"
        training_data_path = f"{CUR_DIR}/datasets/lrnn/{domain_name}/classic/{target_subdir}"

        if not opts.do_not_json:
            # (Fast) convert raw pddl data to json containing state space info
            with TimerContextManager("converting PDDL data to JSON") as timer:
                convert_to_json(domain_name)
                total_time += timer.get_time()

        # (Very fast) preprocess json data
        with TimerContextManager("creating LRNN training dataset from JSON") as timer:
            prepare_training_data(
                domain.name,
                target_subdir=target_subdir,
                cur_dir=CUR_DIR,
                state_regression=state_regression,
                action_regression=action_regression,
                add_types=True,
                samples_limit=samples_limit,
            )
            # if generation of a new limited dataset is requested, we further train in full on it
            samples_limit = -1
            total_time += timer.get_time()

        # Main training function located here!
        with TimerContextManager("training the policy template") as timer:
            policy.train_model_from(
                training_data_path,
                samples_limit=samples_limit,
                num_epochs=num_epochs,
                state_regression=state_regression,
                action_regression=action_regression,
                aggregations=aggregation,
                load_built_samples=opts.cache,
                eval_bk_policy=opts.eval_bk_policy,
            )
            total_time += timer.get_time()

        # save trained model if requested
        if save_file_name:
            policy.store_policy(save_file_name)

        # do not run a policy after training
        exit(0)

    """ 2. handle problem information """
    with TimerContextManager(f"loading PDDL problem file") as timer:
        print(f"{test_problem_path=}")
        problem = pymimir.ProblemParser(test_problem_path).parse(domain)
        initial_state = problem.create_state(problem.initial)
        goal = problem.goal
        policy.setup_test_problem(problem)
        total_time += timer.get_time()

    """ 3. run policy """
    policy_executor = PolicyExecutor(policy, initial_state, goal, run_baseline, opts)
    with TimerContextManager("executing policy") as timer:
        plan = policy_executor.execute()


if __name__ == "__main__":
    main()
