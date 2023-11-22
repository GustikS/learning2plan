import neuralogic
from neuralogic.core import R
from neuralogic.dataset import Dataset
from torch_geometric.data import DataLoader

from logic import DomainLanguage, Atom, Object, Predicate


class PlanningState:
    domain: DomainLanguage

    label: int

    atoms: [Atom]  # all atoms
    propositions: [Atom]  # zero arity atoms
    relations: [Atom]  # >=2 arity atoms

    object_properties: {Object: [Predicate]}  # unary atoms

    def __init__(self, domain: DomainLanguage, atoms: [Atom], label: int = -1):
        self.domain = domain
        self.label = label
        self.atoms = atoms

        self.propositions = []
        self.relations = []
        self.object_properties = {}

        self.update(atoms)

    def update(self, atoms: [Atom]):
        for atom in atoms:
            if atom.predicate.arity == 0:
                self.propositions.append(atom)
            elif atom.predicate.arity == 1:
                self.object_properties.setdefault(atom.terms[0], []).append(atom.predicate)
            elif atom.predicate.arity >= 2:
                self.relations.append(atom)
                for term in atom.terms:
                    self.object_properties.setdefault(term, [])  # log the objects even if there are no properties

    @staticmethod
    def parse(domain: DomainLanguage, label_line: str, facts_lines: [str]):
        label = int(label_line)
        facts: [Atom] = []
        for fact_line in facts_lines:
            fact = domain.parse_atom(fact_line)
            facts.append(fact)
        state = PlanningState(domain, facts, label)
        return state


# %%

class Action:
    name: str

    domain: DomainLanguage

    parameter_types: [str]  # term types

    preconditions: [Atom]
    add_effects: [Atom]
    delete_effects: [Atom]

    def __init__(self, name: str, domain: DomainLanguage, parameters: [str], preconditions: [str], add_effects: [str],
                 delete_effects: [str]):
        self.name = name
        self.domain = domain
        self.parameter_types = [self.domain.types[int(par.split(" ")[1])] for par in parameters]
        self.preconditions = [self.parse_atom(precondition) for precondition in preconditions]
        self.add_effects = [self.parse_atom(add_effect) for add_effect in add_effects]
        self.delete_effects = [self.parse_atom(delete_effect) for delete_effect in delete_effects]

    def parse_atom(self, int_line: str) -> Atom:
        ints = [int(i) for i in int_line.split(" ")]
        predicate = self.domain.predicates[ints[0]]
        arguments = ["X" + str(arg) for arg in ints[1:]]  # arguments are just variable indices
        atom = Atom(predicate, arguments)
        return atom


class PlanningDataset:
    name: str

    domain: DomainLanguage

    static_facts: [Atom]
    actions: [Action]
    goal: [Atom]

    states: [PlanningState]

    def __init__(self, name, domain: DomainLanguage, static_facts: [Atom], actions: [Action], goal: [Atom],
                 states: [PlanningState]):
        self.name = name
        self.domain = domain

        self.static_facts = static_facts
        self.actions = actions
        self.goal = goal

        self.states = states

    def enrich_states(self, add_types=True, add_facts=True, add_goal=False):
        for state in self.states:
            if add_facts:
                state.update(self.static_facts)
            if add_types:
                for obj, properties in state.object_properties.items():
                    properties.extend(state.domain.object_types[obj])
            # if add_goal:
            # # todo

    def get_samples(self, structure_class: object.__class__):
        return [structure_class(sample) for sample in self.states]
