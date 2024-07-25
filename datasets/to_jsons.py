#!/usr/bin/env python3

import json
import os

import pymimir
from tqdm import tqdm

# otherwise take too long
MAX_TOTAL_STATES_PER_PROBLEM = 10000

# to terminate faster based on what we see from other domains
MAX_TOTAL_STATES = {
    "blocksworld": 26000,
    "ferry": 11000,
    "satellite": 19000,
    "transport": 17000,
}

current_dir = os.path.dirname(os.path.abspath(__file__))


def get_nonstatic_predicates(domain: pymimir.Domain):
    # nonstatic predicates are the ones that appear in action effects
    # static predicates do not appear in action effects
    predicates = {}
    for schema in domain.action_schemas:
        for effect in schema.effect:
            atom = effect.atom
            predicate = atom.predicate
            name = predicate.name
            if name not in predicates:
                predicates[name] = predicate
            else:
                assert predicates[name] == predicate
    return predicates


def convert_to_json(domain_name):
    n_states = 0

    domain_pddl = f"{current_dir}/pddl/{domain_name}/domain.pddl"
    domain = pymimir.DomainParser(domain_pddl).parse()
    # nonstatic_predicates = get_nonstatic_predicates(domain)
    # static_predicates = []  # TODO

    data = {
        "predicates": {pred.name: pred.arity for pred in domain.predicates},
        "functions": {},
        "schemata": {schema.name: schema.arity for schema in domain.action_schemas},
        "problems": [],
    }

    problems = []
    for i in range(1, 10):
        problems.append(f"p0{i}")
    for i in range(10, 100):
        problems.append(f"p{i}")
    pbar = tqdm(problems)
    for problem_name in pbar:
        problem_pddl = f"{current_dir}/pddl/{domain_name}/training/{problem_name}.pddl"
        if not os.path.exists(problem_pddl):
            continue
        if n_states >= MAX_TOTAL_STATES[domain_name]:
            break

        problem = pymimir.ProblemParser(problem_pddl).parse(domain)

        successor_generator = pymimir.GroundedSuccessorGenerator(problem)
        state_space = pymimir.StateSpace.new(problem, successor_generator, max_expanded=MAX_TOTAL_STATES_PER_PROBLEM)
        if state_space is None:
            # print('Too many states. Break.')
            # break
            continue

        boolean_goals = []
        for goal in problem.goal:
            if goal.negated:
                raise NotImplementedError("negated goals not supported")
            boolean_goals.append(goal.atom.get_name())

        # sort to make it deterministic
        objects = sorted(problem.objects, key=lambda o: o.name)
        problem_data = {
            "problem_pddl": os.path.basename(problem_pddl),
            "objects": [o.name for o in objects],
            "type": [o.type.name for o in objects],
            "static_facts": [],
            "static_fluents": {},
            "boolean_goals": boolean_goals,
            "numeric_goals": [],
            "states": [],
        }

        # Compute optimal actions from h* of states
        for state in state_space.get_states():
            if n_states >= MAX_TOTAL_STATES[domain_name]:
                break
            if state_space.is_goal_state(state):
                continue
            this_h = state_space.get_distance_to_goal_state(state)
            applicable_actions = successor_generator.get_applicable_actions(state)
            action_values = {}
            for action in applicable_actions:
                next_state = action.apply(state)
                h = state_space.get_distance_to_goal_state(next_state)
                action_values[action.get_name()] = h
            if action_values:
                best_h = min(action_values.values())
                best_actions = [key for key, value in action_values.items() if value == best_h]
            else:
                best_actions = []
            n_states += 1
            pbar.set_description(f"{domain_name} {problem_name} {n_states}")

            state_data = {
                "facts": [atom.get_name() for atom in state.get_atoms()],
                "fluents": {},
                "h": this_h,
                "optimal_actions": best_actions,
                "action_values": action_values,
            }
            problem_data["states"].append(state_data)
        data["problems"].append(problem_data)
    print(f"{n_states} states collected from state spaces with <{MAX_TOTAL_STATES_PER_PROBLEM} states.")

    os.makedirs(f"{current_dir}/jsons/{domain_name}/classic", exist_ok=True)
    json_path = f"{current_dir}/jsons/{domain_name}/classic/state_space_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    for domain_name in MAX_TOTAL_STATES:
        convert_to_json(domain_name)


if __name__ == "__main__":
    main()
