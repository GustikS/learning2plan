""" Make use of neuralogic and pymimir to encode policies as Horn clause rules"""
import time
from abc import abstractmethod
from itertools import product
from typing import Union

from neuralogic.core import C, R, Template, V
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.inference.inference_engine import InferenceEngine
from pymimir import ActionSchema, Atom, Domain, Literal

Schema = Union[str, ActionSchema]

## TODO: add typing from pddl files

class Policy:
    def __init__(self, domain: Domain):
        self._domain = domain
        self._schemata = self._domain.action_schemas
        self._predicates = self._domain.predicates
        self._name_to_schema = {schema.name: schema for schema in self._schemata}

    def solve(self, state: list[Atom], goal: list[Literal]) -> list[str]:
        """ given a state and goal pair, return possible actions from policy rules """
        self._template = Template()
        self._add_state(state, goal)
        self._add_predicate_copies()
        self.add_policy_rules()

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

    @abstractmethod
    def add_policy_rules(self):
        raise NotImplementedError
    
    def _add_predicate_copies(self) -> None:
        """ add ug, ag, ap copies of predicates and rules """
        ## do not add ug as it is unachieved
        for prefix, predicate in product(["ag", "ap"], self._predicates):
            variables = [V.get(f"X{i}") for i in range(predicate.arity)]
            self._template += R.get(f"{prefix}_{predicate.name}")(variables) >= R.get(predicate.name)(variables)
    
    def relation_from_schema(self, schema: Schema) -> BaseRelation:
        if isinstance(schema, str):
            schema = self._name_to_schema[schema]
        parameters = [p.name.replace("?", "").upper() for p in schema.parameters]
        parameters = [V.get(p) for p in parameters]
        head = R.get(schema.name)(parameters)
        return head

    def get_schema_preconditions(self, schema: Schema) -> list[BaseRelation]:
        if isinstance(schema, str):
            schema = self._name_to_schema[schema]
        parameters = [p.name for p in schema.parameters]
        parameter_remap = {p: p.replace("?", "").upper() for p in parameters}
        body = []
        for p in schema.precondition:
            toks = p.atom.get_name().replace(" ", "").split("(")
            assert len(toks) <= 2
            predicate = toks[0]
            objects = toks[1].replace(")", "").split(",")
            prec_vars = [V.get(parameter_remap[a]) for a in objects]
            if p.negated:
                # won't be necessary in the next release...
                self._template += R.get(f"n_{predicate}")(prec_vars) <= R.get(predicate)(prec_vars)
                literal = ~R.get(f"n_{predicate}")(prec_vars)
            else:
                literal = R.get(predicate)(prec_vars)
            body.append(literal)
        return body
    
    def _add_state(self, state: list[Atom], goal: list[Literal]) -> None:
        state = set([atom.get_name() for atom in state])
        pos_goals = set()
        neg_goals = set()
        for g in goal:
            if g.negated:
                neg_goals.add(g.atom.get_name())
            else:
                pos_goals.add(g.atom.get_name())
        if len(neg_goals):
            raise NotImplementedError("Negative goals are not supported yet")
        
        atoms_by_type = {
            "ap": state - pos_goals,
            "ag": pos_goals.intersection(state),
            "ug": pos_goals - state
        }

        for prefix, atoms in atoms_by_type.items():
            for atom in atoms:
                toks = atom.split("(")
                assert len(toks) <= 2
                predicate = toks[0]
                predicate = f"{prefix}_{predicate}"
                objects = toks[1].replace(")", "").replace(" ", "").split(",")
                objects = [C.get(obj) for obj in objects]
                fact = R.get(predicate)(objects)
                self._template += [fact]

    def add_facts(self, facts: list[BaseRelation]) -> None:
        for fact in facts:
            self._template.add_rule(fact)

    def add_rules(self, rules: list[BaseRelation]) -> None:
        for rule in rules:
            self._template.add_rule(rule)
