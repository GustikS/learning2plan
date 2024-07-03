import logging
import time

import neuralogic

if not neuralogic.is_initialized():
    neuralogic.initialize()

from neuralogic.core import R

import jpype.imports
import java.util.ArrayList as jList

# direct access to the java backend classes
jLiteral = jpype.JClass("cz.cvut.fel.ida.logic.Literal")
jState = jpype.JClass("cz.cvut.fel.ida.logic.grounding.planning.State")
jAction = jpype.JClass("cz.cvut.fel.ida.logic.grounding.planning.Action")

import pddl
from pddl.logic import Variable, Predicate
from pddl.logic.functions import NumericFunction, NumericValue
from pddl.logic.predicates import EqualTo


def parse_literal(string_literal):
    negated = string_literal.startswith("!")
    split = string_literal.split("(")
    predicate = split[0]
    terms = split[1][:-1].split(",")
    terms = [term.strip() for term in terms]
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

        self.query = R.get(name)(parameters)  # turn action headers into lifted queries
        self.jAction = None

    def backend(self):
        self.jAction = jAction(self.name,
                               jList(self.parameters),
                               jList(self.preconditions),
                               jList(self.add_effects),
                               jList(self.del_effects))
        return self.jAction

    def ground(self, terms):
        if self.jAction is None:
            self.backend()
        return self.jAction.grounding(terms)

    def to_rule(self, predicate_prefix="", dim=3):
        body = []
        for precondition in self.preconditions:
            negated, predicate, terms = parse_literal(precondition)
            body.append(get_literal(f'{predicate_prefix}{predicate}', terms, negated, string=False)[dim, dim])
        head = R.get(self.name)(self.parameters)[1, dim]
        return head <= body


class State:
    def __init__(self, atoms):
        self.atoms = atoms
        self.atoms_ILG = atoms  # to be updated when goal is presented
        self.jState = None

    def backend(self):
        self.jState = jState(",".join(self.atoms))
        return self.jState

    @staticmethod
    def from_grounding(grounding_sample):
        clause = grounding_sample.grounding.groundingWrap.example.clause
        clauseE = grounding_sample.grounding.groundingWrap.example.clauseE
        return jState(clause, clauseE)

    @staticmethod
    def from_backend(state):
        return State(set([str(l) for l in state.clause.literals()]))

    def to_clause(self):
        atoms = []
        for atom in self.atoms:
            negated, predicate, terms = parse_literal(atom)
            atoms.append(get_literal(predicate, terms, negated, string=False))
        return jList(atoms)

    def get_relations(self):
        example = []
        for atom in self.atoms_ILG:
            negated, predicate, terms = parse_literal(atom)
            example.append(R.get(predicate)(terms))
        return example

    def is_goal(self, goal_atoms: set):
        return goal_atoms.issubset(self.atoms)

    def setup_ILG(self, goal_state_atoms):
        """Split the state representation into a pure and ILG-modified representation for an easier use"""
        is_ilg = False
        pure_atoms = []
        ilg_atoms = []
        for atom in self.atoms:
            if atom[:3] in ['ap_', 'ag_']:
                is_ilg = True
                pure_atoms.append(atom[3:])
                ilg_atoms.append(atom)
            elif atom[:3] == 'ug_':
                is_ilg = True
                ilg_atoms.append(atom)
            else:
                pure_atoms.append(atom)
                is_ilg = False
                if atom in goal_state_atoms:
                    iatom = "ag_" + atom
                else:
                    iatom = "ap_" + atom
                ilg_atoms.append(iatom)
        if not is_ilg:
            for goal in goal_state_atoms:
                if goal not in pure_atoms:
                    ilg_atoms.append("ug_" + goal)

        self.atoms = pure_atoms
        self.atoms_ILG = ilg_atoms


def extract_actions(pddl_domain, just_strings=True):
    actions = []
    for action in pddl_domain.actions:
        name = action.name
        parameters = [parse_term(t) for t in action.parameters]
        precs, conj = _check_symbol(action.precondition)
        preconditions = []
        for lit in (precs if conj else [precs]):
            prec = extract_literal(lit, just_strings)
            if isinstance(prec, list):
                preconditions.extend(prec)
            else:
                preconditions.append(prec)
        effects = []
        if hasattr(action.effect, 'operands'):
            operands = action.effect.operands
        else:
            operands = [action.effect]
        for lit in operands:
            l = extract_literal(lit, just_strings)
            if isinstance(l, list):
                effects.extend(l)
            else:
                effects.append(l)
        actions.append(Action(name, parameters, preconditions, effects))
    return actions


def _check_symbol(element):
    """There is a (awkward) nesting from the PDDL parser if there is a conjunction or negation"""
    if hasattr(element, 'SYMBOL'):
        if element.SYMBOL == 'and':
            return element.operands, "and"
        elif element.SYMBOL == 'not':
            return element.argument, "not"
        elif element.SYMBOL.name == 'GREATER_EQUAL':
            return element.operands, "@geq"
        elif element.SYMBOL.name == 'LESSER_EQUAL':
            return element.operands, "@leq"
        elif element.SYMBOL.name == 'DECREASE':
            return element.operands, "@dec"  # TODO
        elif element.SYMBOL.name == 'INCREASE':
            return element.operands, "@inc"  # TODO
        else:
            raise Exception("Unknown PDDL symbol:" + str(element.SYMBOL))
    elif isinstance(element, Predicate) or isinstance(element, NumericFunction) or isinstance(element, NumericValue):
        return element, ""
    else:
        raise Exception("Unknown PDDL symbol")


def get_terms(element):
    if hasattr(element, "terms"):
        return element.terms
    if hasattr(element, "left"):
        return [element.left, element.right]
    if isinstance(element, tuple):
        return list(element)
    else:
        raise Exception("Unknown PDDL terms")


def extract_literal(pddl_literal, just_strings=True, incr=1):
    element, negated = _check_symbol(pddl_literal)
    if not negated or negated == "not":
        if isinstance(element, NumericValue):
            terms = [str(element)]
        else:
            terms = [parse_term(t) for t in get_terms(element)]
        if isinstance(element, EqualTo) or isinstance(element, NumericValue):
            name = "@eq"
        else:
            name = element.name
        if isinstance(element, NumericFunction) or isinstance(element, NumericValue):
            # introducing an auxiliary variable to get rid of the function symbols with predicates
            terms.append(f"XX{incr}")
            if isinstance(element, NumericFunction):
                name = f'has_{element.name}'

    else:
        literals = [extract_literal(t, just_strings, incr=i) for i, t in enumerate(get_terms(element))]
        literals.append(get_literal(negated, [f'XX{i}' for i in range(len(literals))], False, just_strings))
        return literals

    return get_literal(name, terms, negated, just_strings)


def get_literal(predicate_name, terms, negated, string=True):
    if string:
        negation = "!" if negated else ""
        return f'{negation}{predicate_name}({",".join(terms)})'
    else:
        if negated:
            return ~R.get(predicate_name)(terms)
        else:
            return R.get(predicate_name)(terms)


def parse_term(pddl_term, include_types=False):
    """Returns just string representations of PDDL terms"""
    if pddl_term.type_tags and include_types:
        type = f'{list(pddl_term.type_tags)[0]}:'
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
    # domain = "satellite"
    numeric = False
    problems, predicates, actions = parse_domain(domain, numeric=numeric, encoding="")
    states = flatten_states(problems)
    # all successors from the json states
    successors = all_successors(states, actions)
    # next generations of successor states...
    for i in range(3):
        successors = all_successors(successors[:1000], actions)
