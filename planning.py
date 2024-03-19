from learning2plan.solving.lrnn import jlist, Backend
from learning2plan.logic import DomainLanguage, Atom, Object, Predicate, LogicLanguage

goal_relation_prefix = "goal_"


class PlanningState:
    domain: DomainLanguage

    label: int  # optional

    atoms: [Atom]  # all atoms
    propositions: [Atom]  # zero arity atoms
    relations: [Atom]  # >=2 arity atoms

    object_properties: {Object: [Predicate]}  # unary atoms

    def __init__(self, domain: DomainLanguage, atoms: [Atom], label: int = -1):
        self.domain = domain
        self.label = label

        self.atoms = []
        self.propositions = []
        self.relations = []
        self.object_properties = {}

        self.update(atoms)

    def update(self, atoms: [Atom]):
        self.atoms.extend(atoms)
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

    def to_backend(self, backend: Backend):
        literals = LogicLanguage.to_backend(self.atoms, backend)
        return backend.state(literals)

    @staticmethod
    def from_backend(backend_state, domain: DomainLanguage):
        atoms = domain.from_backend(backend_state.clause.literals())
        return PlanningState(domain, atoms)

    def get_sample(self, structure_class: object.__class__):
        sample = structure_class(self)
        sample.load_state(self)
        return sample

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strings = []
        for atom in self.atoms:
            strings.append(atom.predicate.name + "(" + ",".join([term.name for term in atom.terms]) + ")")
        return ", ".join(strings)


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

        self.backend_action = None

    def parse_atom(self, int_line: str) -> Atom:
        ints = [int(i) for i in int_line.split(" ")]
        predicate = self.domain.predicates[ints[0]]
        arguments = ["X" + str(arg) for arg in ints[1:]]  # arguments are just variable indices
        atom = Atom(predicate, arguments)
        return atom

    def to_backend(self, backend):
        if self.backend_action is not None:
            return self.backend_action
        else:
            preconditions = LogicLanguage.to_backend(self.preconditions, backend)
            add_effects = LogicLanguage.to_backend(self.add_effects, backend)
            delete_effects = LogicLanguage.to_backend(self.delete_effects, backend)
            self.backend_action = backend.action(self.name, preconditions, add_effects, delete_effects)
            return self.backend_action

class GroundAction(Action):

    def __init__(self, groundAction, domain):
        self.domain = domain
        self.name = groundAction.lifted.name

        self.preconditions = domain.from_backend(groundAction.preconditions)
        self.add_effects = domain.from_backend(groundAction.addEffects)
        self.delete_effects = domain.from_backend(groundAction.deleteEffects)


class PlanningInstance:
    name: str

    domain: DomainLanguage

    init: [Atom]
    static_facts: [Atom]
    actions: [Action]
    goal: [Atom]

    goal_predicates: {Predicate: Predicate}  # original -> goal version

    def __init__(self, name, domain: DomainLanguage, static_facts: [Atom], init: [Atom], actions: [Action],
                 goal: [Atom]):
        self.name = name
        self.domain = domain

        self.static_facts = static_facts
        self.init = init
        self.actions = actions
        self.goal = goal

    def to_backend(self, backend):
        static_facts = LogicLanguage.to_backend(self.static_facts, backend)
        init_state = LogicLanguage.to_backend(self.init, backend)
        goal_state = LogicLanguage.to_backend(self.goal, backend)
        actions = jlist([action.to_backend(backend) for action in self.actions])
        return backend.instance(self.name, static_facts, init_state, goal_state, actions)


class PlanningDataset(PlanningInstance):
    states: [PlanningState]

    def __init__(self, name, domain: DomainLanguage, static_facts: [Atom], actions: [Action], goal: [Atom],
                 duplicate_goal_predicates=False, remove_duplicate_states=True):
        super().__init__(name, domain, static_facts, None, actions, goal)

        self.states = []

        if duplicate_goal_predicates:
            self.duplicate_goal_predicates()

        if remove_duplicate_states:
            duplicates = self.get_duplicate_states(remove=True)
            if duplicates:
                print("Duplicate states found:")
                print(duplicates)

        self.domain.update()

    def add_state(self, state: PlanningState):
        # todo enrich the state right here when adding?
        self.states.append(state)

    def enrich_states(self, add_types=True, add_facts=True, add_goal=True):
        for state in self.states:
            if add_facts:
                state.update(self.static_facts)
            if add_types:
                # todo add types also as explicit unary atoms here?
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
