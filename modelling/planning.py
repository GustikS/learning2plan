import logging
import time

import neuralogic

if not neuralogic.is_initialized():
    neuralogic.initialize(jar_path="../jar/NeuraLogic.jar")  # custom backend upgrade (to be included in a new version)

from neuralogic.core import R

import jpype.imports
import java.util.ArrayList as jList

# direct access to the java backend classes
jLiteral = jpype.JClass("cz.cvut.fel.ida.logic.Literal")
jState = jpype.JClass("cz.cvut.fel.ida.logic.grounding.planning.State")
jAction = jpype.JClass("cz.cvut.fel.ida.logic.grounding.planning.Action")

import pddl
from pddl.logic import Variable


def parse_literal(string_literal):
    negated = string_literal.startswith("!")
    split = string_literal.split("(")
    predicate = split[0]
    terms = split[1][:-1].split(",")
    terms = terms if terms[0] else []
    return negated, predicate, terms


class Action:
    """Build an action merely from strings"""

    def __init__(self, name, parameters, preconditions, effects):
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions

        add_effects, del_effects = [], []
        for effect in effects:
            if effect.startswith("!"):
                del_effects.append(effect[1:])
            else:
                add_effects.append(effect)

        self.add_effects = add_effects
        self.del_effects = del_effects

    def backend(self):
        return jAction(self.name,
                       jList(self.parameters),
                       jList(self.preconditions),
                       jList(self.add_effects),
                       jList(self.del_effects))

    def to_rule(self, predicate_prefix):
        body = []
        for precondition in self.preconditions:
            negated, predicate, terms = parse_literal(precondition)
            body.append(get_literal(f'{predicate_prefix}_{predicate}', terms, negated, string=False))
        head = R.get(self.name)(self.parameters)
        return head <= body


class State:
    def __init__(self, atoms):
        self.atoms = atoms

    def backend(self):
        return jState(",".join(self.atoms))

    def to_clause(self):
        atoms = []
        for atom in self.atoms:
            negated, predicate, terms = parse_literal(atom)
            atoms.append(get_literal(predicate, terms, negated, string=False))
        return jList(atoms)


def extract_actions(pddl_domain, just_strings=True):
    actions = []
    for action in pddl_domain.actions:
        name = action.name
        parameters = [parse_term(t) for t in action.parameters]
        precs, conj = _check_symbol(action.precondition)
        preconditions = [extract_literal(lit, just_strings) for lit in (precs if conj else [precs])]
        effects = [extract_literal(lit, just_strings) for lit in action.effect.operands]
        actions.append(Action(name, parameters, preconditions, effects))
    return actions


def _check_symbol(element):
    """There is a (awkward) nesting from the PDDL parser if there is a conjunction or negation"""
    if hasattr(element, 'SYMBOL'):
        if element.SYMBOL == 'and':
            return element.operands, "and"
        elif element.SYMBOL == 'not':
            return element.argument, "not"
        else:
            raise Exception("Unknown PDDL symbol")
    else:
        return element, ""


def extract_literal(pddl_literal, just_strings=True):
    element, negated = _check_symbol(pddl_literal)
    terms = [parse_term(t) for t in element.terms]
    return get_literal(element, terms, negated, just_strings)


def get_literal(predicate, terms, negated, string=True):
    if string:
        negation = "!" if negated else ""
        return f'{negation}{predicate.name}({",".join(terms)})'
    else:
        if negated:
            return ~R.get(predicate)(terms)
        else:
            return R.get(predicate)(terms)


def parse_term(pddl_term):
    """Returns just string representations of PDDL terms"""
    if pddl_term.type_tags:
        type = f'{pddl_term.type_tags[0]}:'
    else:
        type = ""
    if isinstance(pddl_term, pddl.logic.Variable):
        term = f'{type}{pddl_term.name.title()}'  # capitalize variable names
    else:
        term = f'{type}{pddl_term.name.lower()}'  # constants are lower-case
    return term


def all_successors(states, actions):
    """Exhaustively generates all possible successors of all the states through all the actions"""
    logging.log(logging.INFO, f'number of loaded states: {len(states)}')
    if isinstance(states[0], tuple):
        time_start = time.time()
        jstates = [State(state[0]).backend() for state in states]
        logging.log(logging.INFO, f"Java parsing of the states: [{time.time() - time_start:.1f} seconds]")
    else:  # states already parsed
        jstates = states

    time_start = time.time()
    jactions = [action.backend() for action in actions]
    logging.log(logging.INFO, f"Java parsing of the actions: [{time.time() - time_start:.1f} seconds]")

    time_start = time.time()
    all_successors = []
    for state in jstates:
        for action in jactions:
            successors = state_successors(state, action)
            all_successors.extend(successors)

    logging.log(logging.INFO,
                f"Generating all possible {len(all_successors)} successors from the"
                f" previous {len(states)} states times {len(actions)} actions: {time.time() - time_start:.1f} seconds")
    return all_successors


def state_successors(state, action):
    """Just a single state and one (lifted) action"""
    successors = []
    logging.log(logging.DEBUG, f'state:  {state}')
    substitutions = action.substitutions(state)
    logging.log(logging.DEBUG, f'substitutions:\n{substitutions}')
    ground_actions = action.groundings(substitutions)
    logging.log(logging.DEBUG, f'successors:\n')
    for ground_action in ground_actions:
        successor = ground_action.successor(state)
        logging.log(logging.DEBUG, f'{ground_action} -> {successor}')
        successors.append(successor)
    return successors


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        # level = logging.INFO,
        format=''
    )
    from modelling.samples import parse_domain, flatten_states

    domain = "blocksworld"
    numeric = False
    problems, predicates, actions = parse_domain(domain, numeric=numeric, encoding="")
    states = flatten_states(problems)
    # all successors from the json states
    successors = all_successors(states, actions)
    # next generations of successor states...
    for i in range(3):
        successors = all_successors(successors[:1000], actions)
