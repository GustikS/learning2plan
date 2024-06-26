import json
import logging
import os
import sys

import neuralogic
from neuralogic.logging import Formatter, Level, add_handler

from modelling.planning import extract_actions, Action


# add_handler(sys.stdout, Level.FINE, Formatter.COLOR)

# from pddl import parse_domain as pddl_parse

def get_filename(domain_name, numeric, format, path, filename):
    version = "numeric" if numeric else "classic"
    assert format in ["jsons", "lrnn"]
    file_path = f"{path}/datasets/{format}/{domain_name}/{version}/{filename}"
    return file_path


def parse_domain(domain, numeric=False, encoding="ILG", problem_limit=-1):
    json_data, pddl_domain = load_file(domain, numeric=numeric)
    problems, predicates, actions = parse_json(json_data, encoding=encoding, problem_limit=problem_limit)
    if pddl_domain is not None:
        actions = extract_actions(pddl_domain)  # replace the action names with their full specifications
    return problems, predicates, actions


def load_file(domain_name, numeric=False, path="../"):
    logging.log(logging.INFO, "loading domain")

    json_file_path = get_filename(domain_name, numeric, "jsons", path, "data.json")

    with open(json_file_path, 'r') as f:
        json_data = json.loads(f.read())

    try:
        domain = parse_pddl(domain_name, numeric, path)
    except:
        domain = None

    return json_data, domain


def parse_pddl(domain_name, numeric=False, path="../"):
    filename = get_filename(domain_name, numeric, "jsons", path, "domain.pddl")
    return pddl_parse(filename)


# TODO transform all the flags here into a class hierarchy of possible state encodings (reusing the existing classes...)
def parse_json(json_data, problem_limit=-1, state_limit=-1, merge_static=True,
               encoding="ILG", logic_numbers=False, add_objects=False):
    logging.log(logging.INFO, "parsing domain")
    actions = json_data['schemata']  # to work with these I'd also need their preconditions...

    actions = [Action(name, [f'X{i}' for i in range(arity)], None, []) for name, arity in actions.items()]

    functions = encode_functions(json_data['functions'], logic_numbers)
    predicates = encode_predicates(json_data['predicates'], encoding)
    predicates.update(functions)  # I think these are just the same thing?
    is_numeric = len(functions) > 0

    problems = {}
    for problem in json_data['problems'][:problem_limit]:
        file = problem['problem_pddl']

        object_names = encode_objects(problem['objects'])
        static_facts = set(problem['static_facts'])
        static_fluents = set(encode_fluents(problem['static_fluents'], logic_numbers))
        boolean_goals = set(problem['boolean_goals'])
        numeric_goals = encode_fluents(problem['numeric_goals'], logic_numbers)  # just fluents or some constraints?

        states = {}
        for state in problem['states'][:state_limit]:
            h = state["h"]
            if h is None:
                continue  # skip states which we do not know the optimal cost to go

            facts = set(state['facts'] + encode_fluents(state['fluents'], logic_numbers))
            if merge_static:  # add also static facts and fluents
                facts = facts | static_facts | static_fluents

            updated_facts = add_goal_info(facts, boolean_goals, is_numeric, encoding)
            if add_objects:
                updated_facts += object_names

            states[tuple(updated_facts)] = encode_query(h, state["optimal_action"], actions)

        problems[file] = states, boolean_goals

    return problems, predicates, actions


def encode_query(h, optimal_action, all_actions, regression=False):
    if regression:
        return [f'{h} distance']
    else:  # the action classification (or custom loss) mode
        queries = []
        items = optimal_action[1:-1].split(" ")
        queries.append(f'1 {items[0]}({",".join(items[1:])})')  # the target action with a positive label 1
        for action in all_actions:
            queries.append(f'0 {action.name}({",".join(action.parameters)})')  # other actions with label 0
            # Note that this will include the same action with label 0 as well, but these will get aggregated via MAX (by default) in the backend
        return queries


def add_goal_info(facts, boolean_goals, is_numeric_problem, encoding="ILG"):
    ag_facts = facts.intersection(boolean_goals)
    ap_facts = facts.difference(boolean_goals)
    ug_facts = boolean_goals.difference(facts)
    updated_facts = []
    if encoding == "ILG":  # new predicate copies
        for desc, fact_group in [("ag", ag_facts), ("ap", ap_facts), ("ug", ug_facts)]:
            for fact in fact_group:
                if is_numeric_problem:
                    value, fact = split_value(fact)
                    updated_facts.append(f"<{value}> {desc}_{fact}")
                else:
                    updated_facts.append(f"{desc}_{fact}")
    elif encoding == "numeric":  # just a numeric flag of the same info (no copies)
        if is_numeric_problem:
            for ag in ag_facts:
                value, fact = split_value(ag)
                updated_facts.append(f'<[1, 1, {value}]> {fact}')
            for ag in ap_facts:
                value, fact = split_value(ag)
                updated_facts.append(f'<[1, 0, {value}]> {fact}')
            for ag in ug_facts:
                value, fact = split_value(ag)
                updated_facts.append(f'<[0, 1, {value}]> {fact}')
        else:
            updated_facts.extend([f'<[1, 1]> {ag}' for ag in ag_facts])
            updated_facts.extend([f'<[1, 0]> {ap}' for ap in ap_facts])
            updated_facts.extend([f'<[0, 1]> {ug}' for ug in ug_facts])
    else:  # we can also leave it as is to handle it more flexibly later in the template...
        updated_facts = list(facts)
        updated_facts.extend([f"goal_{fact}" for fact in boolean_goals])
    return updated_facts


def split_value(atom):
    try:
        split = atom.split(" ")
        value = float(split[0])  # check for numeric/valued facts
        return value, ' '.join(split[1:])
    except:
        return 1, atom


def encode_fluents(fluents, logical=False):
    if not fluents:
        return []
    if logical:  # the logic constant form
        return [f'value_{atom.replace(")", " " + value + ")")}' for atom, value in fluents.items()]
    else:  # the numeric value form
        return [f'{value} {atom}' for atom, value in fluents.items()]


def encode_functions(functions, logical=False):
    if logical:
        return {name: arity + 1 for name, arity in functions.items()}
    else:
        return functions


def encode_objects(objects):
    # adding at least some info about object types might be more sensible here...
    objects = [f"object({o})" for o in objects]
    return objects


def encode_predicates(orig_predicates, encoding="ILG"):
    predicates = {}
    if encoding == "ILG":
        for predicate, arity in orig_predicates.items():
            # similarly to ILG encoding in Defn. 3.1, annotate whether facts are one of
            # - achieved goal     (ag)
            # - achieved nongoal  (ap)
            # - unachieved goal   (ug)
            for desc in ["ag", "ap", "ug"]:
                new_predicate = f"{desc}_{predicate}"
                assert new_predicate not in orig_predicates
                predicates[new_predicate] = arity
    else:  # we can also deal with that later in the template in multiple ways...
        return orig_predicates

    return predicates


def flatten_states(problems):
    flat_states = []
    for states, _ in problems.values():
        for state, queries in states.items():
            flat_states.append((state, queries))
    return flat_states


def export_problems(problems, domain, numeric, examples_file="examples", queries_file="queries"):
    logging.log(logging.INFO, "exporting problems")
    domain_path = get_filename(domain, numeric, "lrnn", "../", "")
    os.makedirs(domain_path, exist_ok=True)

    with open(f'{domain_path}/{examples_file}.txt', 'w') as e, open(f'{domain_path}/{queries_file}.txt', 'w') as q:
        i = 0
        for states, _ in problems.values():
            for state, queries in states.items():
                e.write(f's{i} :- {", ".join(state)}.\n')
                q.write(f's{i} :- {", ".join(queries)}.\n')
                i += 1

    return domain_path


if __name__ == "__main__":
    domain = "blocksworld"
    # domain = "satellite"
    numeric = False

    problems, predicates, actions = parse_domain(domain, numeric=numeric, encoding="")
    export_problems(problems, domain, numeric=numeric)
