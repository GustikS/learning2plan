from collections import namedtuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from learning2plan.solving.lrnn import jlist

# todo replace with the backend types
Object = namedtuple("Object", "name, type, index")
Predicate = namedtuple("Predicate", "name, arity, types, index")
Atom = namedtuple("Atom", "predicate, terms")


class LogicLanguage:
    objects: [Object]
    predicates: [Predicate]

    object_names: {}
    predicate_names: {}

    types: {int: str}
    supertypes: {str: str}  # type -> supertype

    def __init__(self, objects: [str], predicates: [str], types: [str] = []):
        self.types = {int(obj_type[0]): obj_type[2] for obj_type in types}
        self.types[-1] = "root"  # default
        self.supertypes = {obj_type: self.types[int(parent)] for i, parent, obj_type in types}

        self.objects = [Object(obj_name, self.types[int(obj_type)], int(i)) for i, obj_type, obj_name in objects]
        self.object_names = {obj.name: obj for obj in self.objects}

        self.predicates = []
        for pred in predicates:
            pred_types = [self.types[int(arg_type)] for arg_type in pred[1:-1]]
            pred_name = pred[-1]
            predicate = Predicate(pred_name, len(pred_types), tuple(pred_types), int(pred[0]))
            self.predicates.append(predicate)
        self.predicate_names = {pred.name: pred for pred in self.predicates}

    def parse_atom(self, int_line: str) -> Atom:
        ints = [int(i) for i in int_line.split(" ")]
        predicate = self.predicates[ints[0]]
        constants = tuple([self.objects[i] for i in ints[1:]])
        atom = Atom(predicate, constants)  # no index just yet
        return atom

    @staticmethod
    def to_backend(atoms: [Atom], backend):
        literals = jlist([])  # todo speedup by replacing for the backend types in advance
        for atom in atoms:
            terms = []
            for term in atom.terms:
                if isinstance(term, Object):
                    terms.append(backend.constant.construct(term.name))
                else:
                    terms.append(backend.variable.construct(term))
            termlist = jlist(terms)
            literals.append(backend.literal(atom.predicate.name, termlist))
        return literals

    def from_backend(self, literals) -> [Atom]:
        atoms = []
        if literals:
            for literal in literals:
                atoms.append(Atom(self.predicate_names[literal.predicate().name],
                                  [self.object_names[term.name()] for term in literal.arguments()]))
        return atoms


class DomainLanguage(LogicLanguage):
    """Class with a few useful extras over the pure LogicLanguage"""

    arities: {int: [Predicate]}
    max_arity: int

    nullary_predicates: [Predicate]  # zero arity predicates
    unary_predicates: [Predicate]  # unary relations and (optionally) types
    nary_predicates: [Predicate]  # all other relations with arity >=2 will be treated as relation (edge) types

    object_types: {Object: [Predicate]}  # concrete object types and supertypes for concrete objects

    def __init__(self, objects: [str], predicates: [str], types: [str] = []):
        super().__init__(objects, predicates, types)

    def update(self, types_as_predicates=True):

        self.arities = {}
        self.max_arity = -1
        for predicate in self.predicates:
            self.arities.setdefault(predicate.arity, []).append(predicate)
            if predicate.arity > self.max_arity:
                self.max_arity = predicate.arity

        self.nullary_predicates = self.arities[0] if 0 in self.arities else []
        self.unary_predicates = self.arities[1] if 1 in self.arities else []

        if types_as_predicates:
            pred_index = len(self.predicates)
            type_predicates = {obj_type: Predicate(obj_type, 1, -1, i + pred_index) for i, obj_type in
                               self.types.items() if obj_type != "root"}
            self.unary_predicates.extend(type_predicates.values())

        self.nary_predicates = []
        for arity, predicates in self.arities.items():
            if arity <= 1: continue
            self.nary_predicates.extend(predicates)

        self.object_types = {}
        for obj in self.objects:
            obj_types = []
            self.recursive_types(obj.type, obj_types)
            if types_as_predicates:
                self.object_types[obj] = [type_predicates[obj_type] for obj_type in obj_types]
            else:
                self.object_types[obj] = obj_types

    def recursive_types(self, obj_type, types):
        if obj_type == "root":
            return
        else:
            types.append(obj_type)
            self.recursive_types(self.supertypes[obj_type], types)
