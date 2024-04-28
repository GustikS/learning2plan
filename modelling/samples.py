import json
import logging
import os


def load_json(domain_name, numeric=False, path="../"):
    logging.log(logging.INFO, "loading json")
    if numeric:
        json_file_path = f"{path}/datasets/jsons/{domain_name}/numeric/data.json"
    else:
        json_file_path = f"{path}/datasets/jsons/{domain_name}/classic/data.json"

    with open(json_file_path, 'r') as f:
        json_data = json.loads(f.read())
    return json_data


# TODO transform all the flags here into a class hierarchy of possible state encodings (reusing the existing classes...)
def parse_domain(json_data, problem_limit=-1, state_limit=-1, merge_static=True,
                 encoding="ILG", logic_numbers=False, add_objects=False):
    logging.log(logging.INFO, "parsing domain")
    actions = json_data['schemata']  # to work with these I'd also need their preconditions...

    functions = encode_functions(json_data['functions'], logic_numbers)
    predicates = encode_predicates(json_data['predicates'], encoding)
    predicates.update(functions)  # I think these are just the same thing?

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

            updated_facts = add_goal_info(facts, boolean_goals, encoding)
            if add_objects:
                updated_facts += object_names

            states[tuple(updated_facts)] = encode_query(h, state["optimal_action"], actions)

        problems[file] = states

    return problems, predicates, actions


def encode_query(h, optimal_action, all_actions, regression=True):
    if regression:
        return [f'{h} distance']
    else:  # the action classification (or custom loss) mode
        queries = []
        items = optimal_action[1:-1].split(" ")
        queries.append(f'1 {items[0]}({",".join(items[1:])})')  # the target action with a positive label 1
        for action, arity in all_actions.items():
            if items[0] == action:
                continue
            queries.append(f'0 {action}({",".join([f"X{ar}" for ar in range(arity)])})')  # other actions with label 0
        return queries


def add_goal_info(facts, boolean_goals, encoding="ILG"):
    ag_facts = facts.intersection(boolean_goals)
    ap_facts = facts.difference(boolean_goals)
    ug_facts = boolean_goals.difference(facts)
    updated_facts = []
    if encoding == "ILG":  # new predicate copies
        for desc, fact_group in [("ag", ag_facts), ("ap", ap_facts), ("ug", ug_facts)]:
            for fact in fact_group:
                updated_facts.append(f"{desc}_{fact}")
    elif encoding == "numeric":  # just a numeric flag of the same info (no copies)
        updated_facts.extend([f'[1 1] {ag}' for ag in ag_facts])
        updated_facts.extend([f'[1 0] {ap}' for ap in ap_facts])
        updated_facts.extend([f'[0 1] {ug}' for ug in ug_facts])
    else:  # we can also leave it as is to handle it more flexibly later in the template...
        updated_facts = facts
        updated_facts.extend([f"goal_{fact}" for fact in boolean_goals])
    return updated_facts


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


def export_problems(problems, domain, path="../datasets/lrnn", examples_file="examples", queries_file="queries"):
    logging.log(logging.INFO, "exporting problems")
    domain_path = f'{path}/{domain}'
    os.makedirs(domain_path, exist_ok=True)

    with open(f'{domain_path}/{examples_file}.txt', 'w') as e, open(f'{domain_path}/{queries_file}.txt', 'w') as q:
        for states in problems.values():
            for state, queries in states.items():
                e.write(f'{", ".join(state)}.\n')
                q.write(f'{", ".join(queries)}.\n')

    return domain_path


if __name__ == "__main__":
    # domain = "blocksworld"
    domain = "satellite"
    json_data = load_json(domain, numeric=True)
    problems, predicates, actions = parse_domain(json_data)
    export_problems(problems, domain)
