import argparse
from pathlib import Path
from pprint import pprint

import pymimir
from neuralogic.core import C, R, Template, V
from policy.handcraft_factory import get_handcraft_policy
from util.printing import print_mat
from util.timer import TimerContextManager

_DEFAULT_DOMAIN = "ferry"
# _DEFAULT_DOMAIN = "satellite"


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
    parser.add_argument(
        "-v", "--verbose", type=int, default=0, help="increase output verbosity"
    )
    args = parser.parse_args()
    domain_path = args.domain_path
    problem_path = args.problem_path
    _DEBUG_LEVEL = args.verbose
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
        print(state_repr(state))
        print("=" * 80)
        print("Goal:")
        print(goal_repr(goal))
        print("=" * 80)

    ## execute policy
    with TimerContextManager("executing policy") as timer:
        while not is_goal_state(state, goal):
            policy_actions = policy.solve(state.get_atoms())

            if len(policy_actions) == 0:
                print("Error: No actions computed and not at goal state!")
                print("Terminating...")
                exit(0)

            applicable_actions = successor_generator.get_applicable_actions(state)
            applicable_actions = {a.get_name(): a for a in applicable_actions}

            matrix_log = []

            if _DEBUG_LEVEL > 1:
                print(f"[Step {len(plan)}]")
                matrix_log.append(["Policy actions", ", ".join(policy_actions)])
            action = applicable_actions[policy_actions[0]]

            if _DEBUG_LEVEL > 0:
                matrix_log.append(["Applying", action.get_name()])
            plan.append(action.get_name())

            state = action.apply(state)
            if _DEBUG_LEVEL > 1:
                ilg_state = policy.get_ilg_facts(state.get_atoms())
                ilg_state = ", ".join([str(f) for f in ilg_state])
                matrix_log.append(["Current state", ilg_state])

            print_mat(matrix_log, rjust=False)

            if _DEBUG_LEVEL > 3:
                breakpoint()

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
