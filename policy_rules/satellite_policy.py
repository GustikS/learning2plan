import argparse
import time
from pathlib import Path
from pprint import pprint

import pymimir
from neuralogic.core import C, R, Template, V
from neuralogic.inference.inference_engine import InferenceEngine
from policy.policy import Policy
from policy.satellite import SatellitePolicy


def satellite_rules():
    pass

def is_goal_state(state: pymimir.State, goal: list[pymimir.Literal]):
    state_atoms = set(state.get_atoms())
    for g in goal:
        if g not in state_atoms and not g.negated:
            return False
        elif g in state_atoms and g.negated:
            return False
    return True

def print_state(state: pymimir.State):
    atom_names = sorted([a.get_name() for a in state.get_atoms()])
    # pprint(atom_names)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem_path", type=str, default="l4np/satellite/classic/testing/p0_01.pddl")
    args = parser.parse_args()
    domain_path = "l4np/satellite/classic/domain.pddl"
    problem_path = args.problem_path
    assert Path(domain_path).exists(), f"Domain file not found: {domain_path}"
    assert Path(problem_path).exists(), f"Problem file not found: {problem_path}"

    domain = pymimir.DomainParser(str(domain_path)).parse()
    problem = pymimir.ProblemParser(str(problem_path)).parse(domain)
    successor_generator = pymimir.LiftedSuccessorGenerator(problem)
    state = problem.create_state(problem.initial)
    print_state(state)
    goal = problem.goal

    policy = SatellitePolicy(domain)

    plan = []
    
    while not is_goal_state(state, goal):
        policy_actions = policy.solve(state.get_atoms(), goal)

        applicable_actions = successor_generator.get_applicable_actions(state)
        applicable_actions = {a.get_name(): a for a in applicable_actions}

        action = applicable_actions[policy_actions[0]]

        print_state(state)
        print(action.get_name())
        plan.append(action.get_name())

        state = action.apply(state)
        breakpoint()


    ## solve; TODO may need typing
    print("="*80)
    ret = policy.solve(state, goal)
    pprint(ret)

    ## mimir successor generator usage
    print("="*80)
    initial_state = problem.create_state(initial_state)
    applicable_actions = successor_generator.get_applicable_actions(initial_state)
    applicable_actions = {a.get_name(): a for a in applicable_actions}


if __name__ == "__main__":
    main()
