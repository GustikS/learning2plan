#!/usr/bin/env python

import argparse
import os
import random
import sys

from neuralogic.nn.java import NeuraLogic

sys.path.append("..")  # just a quick fix for the tests to pass... to be removed

from pathlib import Path

import neuralogic
import numpy as np
import pymimir
from neuralogic.logging import Formatter, Level, add_handler
from termcolor import colored

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

if not neuralogic.is_initialized():
    jar_path = f"{CUR_DIR}/jar/NeuraLogic.jar"
    # custom momentary backend upgrades
    neuralogic.initialize(jar_path=jar_path, debug_mode=False)

from policy_rules.policy.handcraft.handcraft_factory import get_handcraft_policy
from policy_rules.util.printing import print_mat
from policy_rules.util.template_settings import load_stored_model
from policy_rules.util.timer import TimerContextManager


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


def state_repr(state: pymimir.State):
    atom_names = sorted([a.get_name() for a in state.get_atoms()])
    return ", ".join(atom_names)


def goal_repr(goal: list[pymimir.Literal]):
    goals = []
    for g in goal:
        if g.negated:
            goals.append(f"~{g.atom.get_name()}")
        else:
            goals.append(g.atom.get_name())
    return ", ".join(goals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", type=str, default="miconic")
    parser.add_argument("-p", "--problem", type=str, default="0_02", help="Of the form 'x_yy'")
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("-b", "--bound", type=int, default=100, help="Termination bound.")
    parser.add_argument("-t", "--template", type=str, default="", help="Policy template name.")
    parser.add_argument("-f", "--files", type=str, default="", help="Save template file(s) name.")
    parser.add_argument("-l", "--learning", type=str, default="", choices=["", "_", str],
                        help="Training data subdirectory ( '_' for root subdir).")
    parser.add_argument("-e", "--embedding", type=int, default=3, choices=[-1, 1, int],
                        help="Embedding dimensionality throughout the model (-1 = off, 1 = scalar)")
    parser.add_argument("-n", "--layers", type=int, default=1, choices=[-1, 1, int],
                        help="Number of model layers (-1 = off, 1 = just embedding, 2+ = message-passing)")
    parser.add_argument("-k", "--skip", type=bool, default=False,
                        help="An option to skip the domain knowledge and use just a generic ML model")
    parser.add_argument("-s", "--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("-c", "--choice", default="best", choices=["sample", "best"],
                        help="Choose the best action or sample from the policy.")
    args = parser.parse_args()
    random.seed(args.seed)
    neuralogic.manual_seed(args.seed)
    np.random.seed(args.seed)
    domain_name = args.domain
    problem_name = args.problem
    template_name = args.template
    save_file_name = args.files
    training_data_dir = args.learning
    embed_dim = args.embedding
    num_layers = args.layers
    skip_knowledge = args.skip
    domain_path = f"{CUR_DIR}/policy_rules/l4np/{domain_name}/classic/domain.pddl"
    test_problem_path = f"{CUR_DIR}/policy_rules/l4np/{domain_name}/classic/testing/p{problem_name}.pddl"
    template_path = f"{CUR_DIR}/datasets/lrnn/{domain_name}/classic/{template_name}"
    template_saving_path = f"{CUR_DIR}/datasets/lrnn/{domain_name}/classic/{save_file_name}"
    training_data_path = f"{CUR_DIR}/datasets/lrnn/{domain_name}/classic/{training_data_dir}"
    _DEBUG_LEVEL = args.verbose
    assert Path(domain_path).exists(), f"Domain file not found: {domain_path}"
    assert Path(training_data_path).exists() or Path(test_problem_path).exists(), \
        f"No training at: {training_data_path} nor testing at: {test_problem_path} Problem(s) found!"

    # TODO(DZC): cycle checking

    if _DEBUG_LEVEL > 3:
        add_handler(sys.stdout, Level.INFO, Formatter.COLOR)
    else:
        add_handler(sys.stdout, Level.OFF, Formatter.COLOR)

    total_time = 0

    with TimerContextManager(f"parsing PDDL domain file {str(domain_path)}") as timer:
        domain = pymimir.DomainParser(str(domain_path)).parse()
        total_time += timer.get_time()

    # possibly load an initial template from file with the same template_name if found
    loaded_model: NeuraLogic = load_stored_model(template_path) if template_name else None

    policy = get_handcraft_policy(domain.name)(domain, debug=_DEBUG_LEVEL)

    with TimerContextManager("initialising policy template") as timer:
        # no need to recreate the template with every new state, we can retain it for the whole domain
        policy.init_template(loaded_model,
                             dim=embed_dim, num_layers=num_layers,
                             skip_knowledge=skip_knowledge,
                             add_types=not training_data_dir  # don't use typing in training at the moment
                             )
        if _DEBUG_LEVEL > 0:
            policy._debug_template()
        total_time += timer.get_time()

    if training_data_dir and hasattr(policy, "model"):
        with TimerContextManager("training the policy template") as timer:
            policy.train_model_from(training_data_path)
            total_time += timer.get_time()

    if save_file_name:
        policy.store_policy(template_saving_path)

    if problem_name:
        with TimerContextManager(f"parsing + loading PDDL problem file {str(test_problem_path)}") as timer:
            problem = pymimir.ProblemParser(str(test_problem_path)).parse(domain)
            state = problem.create_state(problem.initial)
            goal = problem.goal
            policy.setup_test_problem(problem)
            total_time += timer.get_time()

    # print initial state
    if _DEBUG_LEVEL > 1:
        # may or may not be implemented depending on domain
        policy.print_state(state.get_atoms())

    plan = []

    if _DEBUG_LEVEL > 0:
        print("=" * 80)
        print("Initial state:")
        print(state_repr(state))
        print("=" * 80)
        print("Goal:")
        print(goal_repr(goal))
        print("=" * 80)

    ## execute policy
    with TimerContextManager("executing policy") as timer:
        while True:
            goals_left = goal_count(state, goal)
            if goals_left == 0:
                break

            # returns list[tuple[float, Action]]
            policy_actions = policy.solve(state.get_atoms())

            if len(policy_actions) == 0:
                if _DEBUG_LEVEL > 1:
                    # may or may not be implemented depending on domain
                    policy.print_state(state.get_atoms())
                print("Error: No actions computed and not at goal state!")
                print("Terminating...")
                exit(-1)

            matrix_log = []

            Step = len(plan)
            print(f"[{Step=}, {goals_left=}, {timer.get_time()}s]")
            if _DEBUG_LEVEL > 1:
                action_names = [f'{v}:{a.get_name()}' for v, a in policy_actions]
                matrix_log.append(["Available policy actions", ", ".join(action_names)])

            if args.choice == "sample":
                actions = [a[1] for a in policy_actions]
                p = [a[0] for a in policy_actions]
                div = sum(np.exp(p))
                p = np.exp(p) / div  # softmax
                action = np.random.choice(actions, p=p)
            else:
                sorted_actions = sorted(policy_actions, key=lambda item: item[0], reverse=True)
                action = sorted_actions[0][1]  # select the best action

            if _DEBUG_LEVEL > 0:
                matrix_log.append(["Applying", colored(action.get_name(), "cyan")])
            plan.append(action.get_name())

            state = action.apply(state)
            if _DEBUG_LEVEL > 1:
                ilg_state = policy.get_ilg_facts(state.get_atoms())
                ilg_state = ", ".join([str(f) for f in ilg_state])
                matrix_log.append(["Current state: ", ilg_state])
            if len(matrix_log) > 0:
                print_mat(matrix_log, rjust=False)
            if _DEBUG_LEVEL > 1:
                # may or may not be implemented depending on domain
                policy.print_state(state.get_atoms())

            if _DEBUG_LEVEL > 3:
                breakpoint()

            if len(plan) == args.bound:
                print(f"Terminating with failure after {args.bound} steps. Increase bound with -b <bound>", flush=True)
                exit(-1)

        total_time += timer.get_time()
    plan_length = len(plan)

    print("=" * 80)
    print("Plan generated!")
    for action in plan:
        print(action)
    print(f"{plan_length=}")
    print(f"{total_time=}")
    print("=" * 80)


if __name__ == "__main__":
    main()
