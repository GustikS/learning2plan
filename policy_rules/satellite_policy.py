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
    initial_state = problem.initial
    goal = problem.goal

    policy = SatellitePolicy(domain)

    # ## debug
    # print("="*80)
    # print("Template:")
    # print(policy._template)

    ## solve; TODO may need typing
    print("="*80)
    ret = policy.solve(initial_state, goal)
    pprint(ret)

    ## mimir successor generator usage
    print("="*80)
    initial_state = problem.create_state(initial_state)
    applicable_actions = successor_generator.get_applicable_actions(initial_state)
    applicable_actions = [a.get_name() for a in applicable_actions]
    print(f'actions computed by mimir=')
    pprint(applicable_actions)
    print("="*80)
    # succ = applicable_actions[0].apply(initial_state)
    # print(len(applicable_actions))


if __name__ == "__main__":
    main()
