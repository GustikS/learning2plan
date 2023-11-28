import neuralogic
from neuralogic.core import R
from neuralogic.dataset import Dataset
from torch_geometric.data import DataLoader

from logic import DomainLanguage, Atom, Object, Predicate

goal_relation_prefix = "goal_"


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

    goal_predicates: {Predicate: Predicate}  # original -> goal version

    def __init__(self, name, domain: DomainLanguage, static_facts: [Atom], actions: [Action], goal: [Atom],
                 states: [PlanningState], duplicate_goal_predicates=True):
        self.name = name
        self.domain = domain

        self.static_facts = static_facts
        self.actions = actions
        self.goal = goal

        self.states = states

        if duplicate_goal_predicates:
            self.duplicate_goal_predicates()

        self.domain.update()

    def enrich_states(self, add_types=True, add_facts=True, add_goal=True):
        for state in self.states:
            if add_facts:
                state.update(self.static_facts)
            if add_types:
                for obj, properties in state.object_properties.items():
                    properties.extend(state.domain.object_types[obj])
            if add_goal:
                state.update(self.goal)

    def get_samples(self, structure_class: object.__class__):
        samples = []
        for state in self.states:
            sample = structure_class(state)
            sample.load_state(state)
            samples.append(sample)
        return samples

    def get_duplicate_states(self, remove=True):
        # not very efficient but clear
        duplicates = []
        for state in self.states:
            if self.states.count(state) > 1:
                duplicates.append(state)
                if remove:
                    self.states.remove(state)
        return duplicates

    def duplicate_goal_predicates(self):
        goal_atoms = []
        self.goal_predicates = {}
        for atom in self.goal:
            if atom.predicate in self.goal_predicates:
                goal_predicate = self.goal_predicates[atom.predicate]
            else:
                goal_predicate = Predicate(goal_relation_prefix + atom.predicate.name, atom.predicate.arity,
                                           atom.predicate.types, len(self.domain.predicates))
                self.domain.predicates.append(goal_predicate)
                self.goal_predicates[atom.predicate] = goal_predicate

            goal_atom = Atom(goal_predicate, atom.terms)
            goal_atoms.append(goal_atom)
        self.goal = goal_atoms

    def expand_goal_states(self):
        """The version with adding a context object, i.e. keeping all the objects and relations the same,
        but increasing arity of all the relations in all the states instead"""

        # todo
        pass
