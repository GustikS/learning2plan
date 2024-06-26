import argparse
import sys
from pathlib import Path
from pprint import pprint

import pymimir

import neuralogic
from neuralogic.logging import Formatter, Level, add_handler

if not neuralogic.is_initialized():
    neuralogic.initialize(jar_path="../jar/NeuraLogic.jar", debug_mode=False)  # custom momentary backend upgrades

from policy.handcraft.handcraft_factory import get_handcraft_policy
from util.printing import print_mat
from util.timer import TimerContextManager


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
    parser.add_argument("-d", "--domain", type=str, default="ferry")
    parser.add_argument("-p", "--problem", type=str, default="0_01", help="Of the form 'x_yy'")
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("-b", "--bound", type=int, default=100, help="Termination bound.")
    parser.add_argument("-t", "--template", type=str, default="")
    args = parser.parse_args()
    domain_name = args.domain
    problem_name = args.problem
    template_name = args.template
    domain_path = f"l4np/{domain_name}/classic/domain.pddl"
    problem_path = f"l4np/{domain_name}/classic/testing/p{problem_name}.pddl"
    template_path = f"../datasets/lrnn/{domain_name}/classic/{template_name}"
    _DEBUG_LEVEL = args.verbose
    assert Path(domain_path).exists(), f"Domain file not found: {domain_path}"
    assert Path(problem_path).exists(), f"Problem file not found: {problem_path}"

    if _DEBUG_LEVEL > 4:
        add_handler(sys.stdout, Level.ALL, Formatter.COLOR)
    else:
        add_handler(sys.stdout, Level.OFF, Formatter.COLOR)

    total_time = 0

    with TimerContextManager("parsing PDDL files") as timer:
        domain = pymimir.DomainParser(str(domain_path)).parse()
        problem = pymimir.ProblemParser(str(problem_path)).parse(domain)
        state = problem.create_state(problem.initial)
        goal = problem.goal
        total_time += timer.get_time()

    with TimerContextManager("initialising policy") as timer:
        policy = get_handcraft_policy(domain.name)(domain, problem, template_path, debug=_DEBUG_LEVEL)
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

            policy_actions = policy.solve(state.get_atoms())

            if len(policy_actions) == 0:
                print("Error: No actions computed and not at goal state!")
                print("Terminating...")
                exit(-1)

            matrix_log = []

            Step = len(plan)
            print(f"[{Step=}, {goals_left=}, {timer.get_time()}s]")
            if _DEBUG_LEVEL > 1:
                action_names = [f'{v}:{a.get_name()}' for v, a in policy_actions]
                matrix_log.append(["Policy actions", ", ".join(action_names)])

            sorted_actions = sorted(policy_actions, key=lambda item: item[0], reverse=True)
            action = sorted_actions[0][1]   # select the best action

            if _DEBUG_LEVEL > 0:
                matrix_log.append(["Applying", action.get_name()])
            plan.append(action.get_name())

            state = action.apply(state)
            if _DEBUG_LEVEL > 1:
                ilg_state = policy.get_ilg_facts(state.get_atoms())
                ilg_state = ", ".join([str(f) for f in ilg_state])
                matrix_log.append(["Current state", ilg_state])
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
