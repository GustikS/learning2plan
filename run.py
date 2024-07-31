#!/usr/bin/env python

import argparse
import os
import random
import sys
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
    parser.add_argument("-d", "--domain", type=str, default="blocksworld")
    parser.add_argument("-p", "--problem", type=str, default="0_01", help="Of the form 'x_yy'")
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("-b", "--bound", type=int, default=100, help="Termination bound")

    parser.add_argument("-dnc", "--do_not_cycle_detect", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Do not cycle detect when sampling actions")

    parser.add_argument("--train", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Train an LRNN. Training is also performed if a save file is specified.")
    parser.add_argument("-load", "--load_file", type=str, default="", help="Filename to load the template")
    parser.add_argument("-save", "--save_file", type=str, default="", help="Filename to save the template")

    # Data arguments
    parser.add_argument("-lim", "--limit", type=int, default=-1,
                        help="Training data samples cutoff limit (good for quicker debugging)")
    parser.add_argument("-sr", "--state_regression", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Include state h distance labels for (classic) state regression training")
    parser.add_argument("-ar", "--action_regression", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Switch between regression/classification labels for output actions in training")
    parser.add_argument("-dnj", "--do_not_json", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Do not convert PDDL data to JSON")

    # Model training arguments
    parser.add_argument("-e", "--embedding", type=int, default=1,
                        help="Embedding dimensionality throughout the model (-1 = off, 1 = scalar)")
    parser.add_argument("-gnn", "--gnn_type", type=str, default="SAGE", choices=["SAGE", "GIN", "TAG"])
    parser.add_argument("-num", "--layers", type=int, default=1,
                        help="Number of model layers (-1 = off, 1 = just embedding, 2+ = message-passing)")
    parser.add_argument("-agg", "--aggregation", default="max", choices=["sum", "mean", "max"],
                        help="Aggregation function for message passing")
    parser.add_argument("-ep", "--epochs", type=int, default=100,
                        help="Number of model training epochs")

    parser.add_argument("-k", "--knowledge", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="An option to skip the domain knowledge and use just a generic ML model")
    parser.add_argument("-vis", "--visualise", default=None,
                        help="Save visualisation of template to file.")
    parser.add_argument("-s", "--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("-c", "--choice", default="best", choices=["sample", "best"],
                        help="Choose the best action or sample from the policy. Has no effect for baseline policy which defaults to uniform sampling.")

    # Other debugging options
    parser.add_argument("-eval_bk", "--eval_bk_policy", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Computes percentages from training data for the baseline BK policy")
    parser.add_argument("-ca", "--cache", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Store or use stored built samples. Does not work because java objects are not picklable")

    args = parser.parse_args()
    return args
    # fmt: on


def goal_count(state: pymimir.State, goal: list[pymimir.Literal]) -> int:
    state_atoms = set([a.get_name() for a in state.get_atoms()])
    ret = 0
    for g in goal:
        g_name = g.atom.get_name()
        if g_name not in state_atoms and not g.negated:
            ret += 1
        elif g_name in state_atoms and g.negated:
            ret += 1
    return ret


def state_repr(state: Union[pymimir.State, list[pymimir.Literal]], is_goal=False):
    if not is_goal:
        atom_names = sorted([a.get_name() for a in state.get_atoms()])
        return " ".join(atom_names)
    else:
        goals = []
        for g in state:
            if g.negated:
                goals.append(f"~{g.atom.get_name()}")
            else:
                goals.append(g.atom.get_name())
        return " ".join(goals)


def sample_action(policy_actions, sampling_method):
    actions = [a[1] for a in policy_actions]
    p = [a[0] for a in policy_actions]
    p = np.array(p)
    indices = list(range(len(policy_actions)))
    match sampling_method:
        case "uniform":
            # uniform sampling
            action_index = np.random.choice(indices)
        case "sample":
            # sample from distribution computed by scores
            p = p / sum(p)
            action_index = np.random.choice(indices, p=p)
        # case "sample":
        #     # sample after converting distrubtion with softmax
        #     # p = 1/(1 + np.exp(-p))  # sigmoid so softmax does not overflow
        #     p = p / sum(p)  # normalise so softmax does not overflow
        #     div = sum(np.exp(p))
        #     p = np.exp(p) / div  # softmax
        #     print(p)
        #     action_index = np.random.choice(indices, p=p)
        case "highest":
            # if action classification we take the highest,
            action_index = np.argmax(p)
        case "lowest":
            # if action regression we take the lowest
            action_index = np.argmin(p)
    return action_index


def execute_policy(policy, initial_state, goal, pre_policy_time, baseline_policy, args):
    """Main function for executing the policy"""
    _DEBUG_LEVEL = args.verbose

    plan = []
    total_time = 0
    cycles_detected = 0
    state = initial_state
    seen_states = set()

    if baseline_policy:
        sampling_method = "uniform"
    elif args.choice == "sample":
        sampling_method = "sample"
    elif args.action_regression:
        sampling_method = "lowest"
    else:
        sampling_method = "highest"

    with TimerContextManager("executing policy") as timer:
        while True:
            goals_left = goal_count(state, goal)
            if goals_left == 0:
                break

            Step = len(plan)
            print(colored(f"[[[{Step=}, {goals_left=}, {cycles_detected=}, {timer.get_time()}s]]]", "blue"))

            policy_actions: list[tuple[float, pymimir.Action]] = policy.solve(state.get_atoms())

            # sort for reproducibility
            policy_actions = sorted(policy_actions, key=lambda x: x[1].get_name())

            # print remaining goals
            rem_goals = {}
            ilg_state = policy.get_ilg_facts(state.get_atoms())
            for f in ilg_state:
                pred = f.predicate
                if pred.startswith("ug_"):
                    pred = pred[3:]
                    if pred not in rem_goals:
                        rem_goals[pred] = 0
                    rem_goals[pred] += 1
            desc = ""
            for k, v in rem_goals.items():
                desc += f"{k}={v}, "
            print(colored("Remaining goals:", "light_magenta"), desc)

            # print number of derived actions per schema
            schemata = {k.name: 0 for k in policy._domain.action_schemas}
            for a in policy_actions:
                schema = a[1].schema
                schemata[schema.name] += 1
            desc = ""
            for k, v in schemata.items():
                desc += f"{k}={v}, "
            print(colored("Derived:", "light_magenta"), desc)

            if not args.do_not_cycle_detect:
                state_str = state_repr(state)
                seen_states.add(state_str)
            # print(state_str)

            if len(policy_actions) == 0:
                # if _DEBUG_LEVEL > 1:
                #     # may or may not be implemented depending on domain
                #     policy.print_state(state.get_atoms())
                print(f"Error: At step {len(plan)} but no actions computed and not at goal state!")
                print("Terminating...")
                exit(-1)

            matrix_log = []
            if _DEBUG_LEVEL > 1:
                action_names = [f"{v}:{a.get_name()}" for v, a in policy_actions]
                matrix_log.append(["Available policy actions", ", ".join(action_names)])

            # sample action based on selected criterion
            while len(policy_actions) > 0:
                action_index = sample_action(policy_actions, sampling_method)
                action = policy_actions[action_index][1]

                if isinstance(action, pymimir.Action):
                    succ_state = action.apply(state)
                else:
                    raise NotImplementedError

                # check for cycles
                # NOTE: if all successors lead to seen states, one of them is chosen anyway
                if state_repr(succ_state) in seen_states:
                    if _DEBUG_LEVEL > 0:
                        matrix_log.append([" " * 8, colored(action.get_name(), "cyan") + colored(f" Loop detected, sampling again...", "red")])
                    del policy_actions[action_index]
                    cycles_detected += 1
                else:
                    break

            # # use -c sample to break out of cycles
            # action_index = sample_action(policy_actions, sampling_method)
            # action = policy_actions[action_index][1]
            # succ_state = action.apply(state)
            # if state_repr(succ_state) in seen_states:
            #     cycles_detected += 1

            matrix_log.append(["Applying", colored(action.get_name(), "cyan")])
            plan.append(action.get_name())
            state = succ_state

            if _DEBUG_LEVEL > 4:
                ilg_state = policy.get_ilg_facts(state.get_atoms())
                ilg_state = ", ".join([str(f) for f in ilg_state])
                matrix_log.append(["State after action: ", ilg_state])
            if len(matrix_log) > 0:
                print_mat(matrix_log, rjust=False)
            if _DEBUG_LEVEL > 0:
                # may or may not be implemented depending on domain
                policy.print_state(state.get_atoms())

            # if _DEBUG_LEVEL > 3:
            #     breakpoint()

            if len(plan) == args.bound:
                # fmt: off
                print(f"Terminating with failure after {args.bound} steps. Increase bound with -b <bound>", flush=True)
                # fmt: on
                exit(-1)

        total_time += timer.get_time()
    plan_length = len(plan)

    print("=" * 80)
    print("Plan generated!")
    for action in plan:
        print(action)
    print(f"{plan_length=}")
    print(f"{cycles_detected=}")
    print(f"{total_time=}")
    print(f"{pre_policy_time=}")
    print("=" * 80)
    exit(0)


def main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    neuralogic.manual_seed(seed)
    np.random.seed(seed)
    domain_name = args.domain
    problem_name = args.problem
    load_file_name = args.load_file
    save_file_name = args.save_file
    to_train = ((not load_file_name) and save_file_name) or args.train or args.eval_bk_policy
    samples_limit = args.limit
    state_regression = args.state_regression
    action_regression = args.action_regression
    embed_dim = args.embedding
    gnn_type = args.gnn_type
    num_layers = args.layers
    aggregation = args.aggregation
    num_epochs = args.epochs
    include_knowledge = args.knowledge
    domain_path = f"{CUR_DIR}/datasets/pddl/{domain_name}/domain.pddl"
    test_problem_path = f"{CUR_DIR}/datasets/pddl/{domain_name}/testing/p{problem_name}.pddl"
    training_data_path = f"{CUR_DIR}/datasets/lrnn/{domain_name}/classic/data"
    _DEBUG_LEVEL = args.verbose
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
            gnn_type=gnn_type
        )
        if _DEBUG_LEVEL > 0:
            policy._debug_template(serialise_file=f"{domain_name}_debug.template.txt")
        total_time += timer.get_time()

    # save visualisation of the template if requested
    save_drawing = args.visualise
    if save_drawing is not None:
        policy.model.draw(filename=save_drawing)
        print(colored(f"Saved template visualisation to {save_drawing}", "green"))

    # training should be performed if there are training data AND the policy has learnable parameters/model
    if to_train and hasattr(policy, "model"):
        training_data_path = f"{CUR_DIR}/datasets/lrnn/{domain_name}/classic/data"

        if not args.do_not_json:
            # (Fast) convert raw pddl data to json containing state space info
            with TimerContextManager("converting PDDL data to JSON") as timer:
                convert_to_json(domain_name)
                total_time += timer.get_time()

        # (Very fast) preprocess json data
        with TimerContextManager("creating LRNN training dataset from JSON") as timer:
            prepare_training_data(
                domain.name,
                target_subdir=f"data_{args.layers}_{args.embedding}_{args.seed}",
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
                load_built_samples=args.cache,
                eval_bk_policy=args.eval_bk_policy,
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

    # print initial state
    if _DEBUG_LEVEL > 1:
        # may or may not be implemented depending on domain
        policy.print_state(initial_state.get_atoms())

    if _DEBUG_LEVEL > 0:
        print("=" * 80)
        print("Initial state:")
        print(state_repr(initial_state, is_goal=False))
        print("=" * 80)
        print("Goal:")
        print(state_repr(goal, is_goal=True))
        print("=" * 80)

    """ 3. run policy """
    execute_policy(
        policy=policy,
        initial_state=initial_state,
        goal=goal,
        pre_policy_time=total_time,
        baseline_policy=run_baseline,
        args=args,
    )


if __name__ == "__main__":
    main()
