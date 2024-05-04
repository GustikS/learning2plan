""" Make use of neuralogic and pymimir to encode policies as Horn clause rules"""

import time
from abc import abstractmethod
from itertools import product
from typing import Union

from neuralogic.core import C, R, Template, V
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.inference.inference_engine import InferenceEngine
from pymimir import ActionSchema, Atom, Domain, Literal, Problem

Schema = Union[str, ActionSchema]

## TODO: add typing from pddl files


class Policy:
    def __init__(self, domain: Domain, problem: Problem, debug=0):
        self._domain = domain
        self._schemata = self._domain.action_schemas
        self._predicates = self._domain.predicates
        self._name_to_schema: dict[str, ActionSchema] = {
            schema.name: schema for schema in self._schemata
        }

        self._problem = problem
        self._debug = debug

    def solve(self, state: list[Atom]) -> list[str]:
        """given a state and goal pair, return possible actions from policy rules"""
        self._init_template()
        self._add_state(state)

        if self._debug > 2:
            print("="*80)
            print(self._template)
            print("="*80)

        engine = InferenceEngine(self._template)

        ret = []
        for schema in self._schemata:
            assignments = engine.query(self.relation_from_schema(schema.name))
            param_to_index = {p.name: i for i, p in enumerate(schema.parameters)}
            for assignment in assignments:
                ret_action = schema.name
                objects = [""] * len(schema.parameters)
                for var, val in assignment.items():
                    objects[param_to_index[f"?{var.lower()}"]] = val
                ret_action = f"{ret_action}({', '.join(objects)})"
                ret.append(ret_action)
        return ret

    def _init_template(self):
        self._template = Template()
        self._add_predicate_copies()
        self._add_object_information()
        self._add_derived_predicates()
        self._add_policy_rules()

    @abstractmethod
    def _add_policy_rules(self):
        raise NotImplementedError
    
    @abstractmethod
    def _add_derived_predicates(self):
        raise NotImplementedError

    def _add_predicate_copies(self) -> None:
        """add ug, ag, ap copies of predicates and rules"""
        ## do not add ug as it is unachieved
        for prefix, predicate in product(["ag", "ap"], self._predicates):
            variables = [V.get(f"X{i}") for i in range(predicate.arity)]
            new_predicate = R.get(f"{prefix}_{predicate.name}")(variables)
            og_predicate = R.get(predicate.name)(variables)
            self._template += og_predicate <= new_predicate

    def _add_object_information(self) -> None:
        """add object types to the template"""
        for obj in self._problem.objects:
            assert obj.is_constant()
            # TODO intermediate types?
            self._template += R.get(obj.type.name)(C.get(obj.name))
            # self._template += R.get(obj.type.base.name)(C.get(obj.name))

    def relation_from_schema(self, schema: Schema) -> BaseRelation:
        """ construct a relation object from a schema"""
        if isinstance(schema, str):
            schema = self._name_to_schema[schema]
        parameters = [p.name.replace("?", "").upper() for p in schema.parameters]
        parameters = [V.get(p) for p in parameters]
        head = R.get(schema.name)(parameters)
        return head
    
    def _get_negative_literal(self, predicate: str, variables: list[str]) -> None:
        """ add a negative literal to the template, hack on top of LRNN bug """
        ## won't be necessary in the next release...
        ## TODO fix when next release comes out
        neg = R.get(f"n_{predicate}")(variables)
        pos = R.get(predicate)(variables)
        self._template += neg <= pos
        return ~neg


    def get_schema_preconditions(self, schema: Schema) -> list[BaseRelation]:
        """ construct base body of a schema from its preconditions with typing """
        if isinstance(schema, str):
            schema = self._name_to_schema[schema]
        schema: ActionSchema = schema

        body = []
        param_remap = {
            p.name: p.name.replace("?", "").upper() for p in schema.parameters
        }

        ## add variables and their types
        for param in schema.parameters:
            assert param.is_variable()
            remap = param_remap[param.name]
            object_type = param.type.name
            atom = R.get(object_type)(V.get(remap))
            body.append(atom)

        ## add preconditions
        for p in schema.precondition:
            p: Literal = p
            toks = p.atom.get_name().replace(" ", "").split("(")
            assert len(toks) <= 2
            predicate = p.atom.predicate.name
            objects = p.atom.terms
            prec_vars = [V.get(param_remap[obj.name]) for obj in objects]
            if p.negated:
                literal = self._get_negative_literal(predicate, prec_vars)
            else:
                literal = R.get(predicate)(prec_vars)
            body.append(literal)

        return body

    def _add_state(self, state: list[Atom]) -> None:
        """ add state information to the template """
        name_to_atom = {atom.get_name(): atom for atom in state}
        pos_goals = set()
        neg_goals = set()
        for g in self._problem.goal:
            if g.negated:
                neg_goals.add(g.atom.get_name())
            else:
                pos_goals.add(g.atom.get_name())
            name_to_atom[g.atom.get_name()] = g.atom
        if len(neg_goals):
            raise NotImplementedError("Negative goals are not supported yet")
        
        state = set([atom.get_name() for atom in state])

        atoms_by_type = {
            "ap": state - pos_goals,
            "ag": pos_goals.intersection(state),
            "ug": pos_goals - state,
        }

        for prefix, atoms in atoms_by_type.items():
            for atom in atoms:
                atom: Atom = name_to_atom[atom]
                predicate = atom.predicate.name
                predicate = f"{prefix}_{predicate}"
                objects = atom.terms
                objects = [C.get(obj.name) for obj in objects]
                if len(objects) == 0:
                    fact = R.get(predicate)()
                else:
                    fact = R.get(predicate)(objects)
                self._template += [fact]

    def add_facts(self, facts: list[BaseRelation]) -> None:
        for fact in facts:
            self._template.add_rule(fact)

    def add_rules(self, rules: list[BaseRelation]) -> None:
        for rule in rules:
            self._template.add_rule(rule)
