#!/usr/bin/env python3

import os

import pymimir
from tqdm import tqdm

DOMAINS = [
    "blocksworld",
    "childsnack",
    "ferry",
    # "floortile",
    "miconic",
    "rovers",
    "satellite",
    "sokoban",
    "spanner",
    "transport",
]

MAX_TOTAL_STATES_PER_PROBLEM = 10000
MAX_TOTAL_STATES_PER_DOMAIN = 25500


def get_static_predicates(domain: pymimir.Domain):
    # todo G->D: what is this supposed to do? Taking effect predicates as static seems wrong... it causes states in the jsons...
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


def main():
    for domain_name in DOMAINS:
        n_states = 0

        domain_pddl = f"ipc23lt/{domain_name}/domain.pddl"
        domain = pymimir.DomainParser(str(domain_pddl)).parse()
        # static_predicates = get_static_predicates(domain) #
        static_predicates = []

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
            problem_pddl = f"ipc23lt/{domain_name}/training/{problem_name}.pddl"
            if not os.path.exists(problem_pddl):
                continue
            if n_states >= MAX_TOTAL_STATES_PER_DOMAIN:
                break

            problem = pymimir.ProblemParser(str(problem_pddl)).parse(domain)

            successor_generator = pymimir.GroundedSuccessorGenerator(problem)
            state_space = pymimir.StateSpace.new(problem, successor_generator,
                                                 max_expanded=MAX_TOTAL_STATES_PER_PROBLEM)
            if state_space is None:
                # print('Too many states. Break.')
                # break
                continue

            static_facts = []
            for atom in problem.initial:
                if atom.predicate.name in static_predicates:
                    static_facts.append(atom.get_name())
            boolean_goals = []
            for goal in problem.goal:
                if goal.negated:
                    raise NotImplementedError("negated goals not supported")
                boolean_goals.append(goal.atom.get_name())

            problem_data = {
                "problem_pddl": os.path.basename(problem_pddl),
                "objects": [o.name for o in problem.objects],
                "static_facts": static_facts,
                "static_fluents": {},
                "boolean_goals": boolean_goals,
                "numeric_goals": [],
                "states": [],
            }

            # best_h = float("inf")
            # best_actions = []   # todo G->D: are the states ordered in decreasing distance manner? Otherwise the best_h check seems suspicious? Check - this indeed seems wrong...
            for state in state_space.get_states():
                if n_states >= MAX_TOTAL_STATES_PER_DOMAIN:
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
                    # if h == 0:
                    #     assert state_space.is_goal_state(next_state)
                    # if h < best_h:
                    #     best_h = h
                    #     best_actions = [action]
                    # elif h == best_h:
                    #     best_actions.append(action)
                if action_values:
                    best_h = min(action_values.values())
                    best_actions = [key for key, value in action_values.items() if value == best_h]
                else:
                    best_actions = []
                n_states += 1
                pbar.set_description(f"{domain_name} {problem_name} {n_states}")

                state_data = {
                    "facts": [atom.get_name() for atom in state.get_atoms() if
                              atom.predicate.name not in static_predicates],
                    "fluents": {},
                    "h": this_h,
                    "optimal_actions": best_actions,
                    "action_values": action_values
                }
                problem_data["states"].append(state_data)
            data["problems"].append(problem_data)
        print(n_states)

        os.makedirs(f"jsons/{domain_name}/classic", exist_ok=True)
        json_path = f"jsons/{domain_name}/classic/state_space_data.json"
        with open(json_path, "w") as f:
            import json
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
