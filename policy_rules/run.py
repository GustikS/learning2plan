import argparse
from pathlib import Path
from pprint import pprint

import pymimir
from policy.handcraft_factory import get_handcraft_policy
from util.timer import TimerContextManager

_DEFAULT_DOMAIN = "ferry"
# _DEFAULT_DOMAIN = "satellite"
_DEBUG_LEVEL = 1


def satellite_rules():
    pass


def is_goal_state(state: pymimir.State, goal: list[pymimir.Literal]):
    state_atoms = set([a.get_name() for a in state.get_atoms()])
    for g in goal:
        g_name = g.atom.get_name()
        if g_name not in state_atoms and not g.negated:
            return False
        elif g_name in state_atoms and g.negated:
            return False
    return True


def print_state(state: pymimir.State):
    atom_names = sorted([a.get_name() for a in state.get_atoms()])
    pprint(atom_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--domain_path",
        type=str,
        default=f"l4np/{_DEFAULT_DOMAIN}/classic/domain.pddl",
    )
    parser.add_argument(
        "-p",
        "--problem_path",
        type=str,
        default=f"l4np/{_DEFAULT_DOMAIN}/classic/testing/p0_01.pddl",
    )
    args = parser.parse_args()
    domain_path = args.domain_path
    problem_path = args.problem_path
    assert Path(domain_path).exists(), f"Domain file not found: {domain_path}"
    assert Path(problem_path).exists(), f"Problem file not found: {problem_path}"

    total_time = 0

    with TimerContextManager("parsing PDDL files") as timer:
        domain = pymimir.DomainParser(str(domain_path)).parse()
        problem = pymimir.ProblemParser(str(problem_path)).parse(domain)
        successor_generator = pymimir.LiftedSuccessorGenerator(problem)
        state = problem.create_state(problem.initial)
        goal = problem.goal
        total_time += timer.get_time()

    with TimerContextManager("initialising policy") as timer:
        policy = get_handcraft_policy(domain.name)(domain, problem, debug=_DEBUG_LEVEL)
        total_time += timer.get_time()

    plan = []

    if _DEBUG_LEVEL > 0:
        print("=" * 80)
        print("Initial state:")
        print_state(state)
        print("=" * 80)
        print("Goal:")
        pprint(goal)
        print("=" * 80)

    ## execute policy
    with TimerContextManager("executing policy") as timer:
        while not is_goal_state(state, goal):
            policy_actions = policy.solve(state.get_atoms())

            applicable_actions = successor_generator.get_applicable_actions(state)
            applicable_actions = {a.get_name(): a for a in applicable_actions}

            action = applicable_actions[policy_actions[0]]

            print(action.get_name())
            plan.append(action.get_name())

            state = action.apply(state)
            if _DEBUG_LEVEL > 2:
                print_state(state)
        total_time += timer.get_time()
    plan_length = len(plan)

    print("=" * 80)
    print("Plan generated!")
    print(f"{plan_length=}")
    print(f"{total_time=}")
    print("=" * 80)


if __name__ == "__main__":
    main()
