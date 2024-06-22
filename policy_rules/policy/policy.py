"""Make use of neuralogic and pymimir to encode policies as Horn clause rules"""

from abc import abstractmethod
from itertools import product
from typing import Union

from neuralogic.core import C, R, Template, V
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.inference.inference_engine import InferenceEngine
from pymimir import Action, ActionSchema, Atom, Domain, Literal, Object, Problem
from util.str_atom import StrAtom

Schema = Union[str, ActionSchema]


class Policy:
    def __init__(self, domain: Domain, problem: Problem, debug=0):
        self._domain = domain
        self._problem = problem
        self._debug = debug
        self._goal = self._problem.goal

        self._schemata = self._domain.action_schemas
        self._predicates = self._domain.predicates
        self._name_to_schema: dict[str, ActionSchema] = {
            schema.name: schema for schema in self._schemata
        }
        self._objects = self._problem.objects
        self._name_to_object: dict[str, Object] = {obj.name: obj for obj in self._objects}

        self._prev_state = None

    def solve(self, state: list[Atom]) -> list[Action]:
        """given a state and goal pair, return possible actions from policy rules"""
        self._init_template()

        ilg_atoms = self.get_ilg_facts(state)

        ## add atoms to template
        for atom in ilg_atoms:
            lrnn_fact = R.get(atom.predicate)([C.get(obj) for obj in atom.objects])
            self._template.add_rule(lrnn_fact)

        if self._debug > 2:
            print("=" * 80)
            print("Template for current state:")
            print(self._template)
            print("=" * 80)

        self._engine = InferenceEngine(self._template)

        if self._debug > 2:
            self._debug_inference()

        ret = []
        for schema in self._schemata:
            assignments = self._engine.query(self.relation_from_schema(schema.name))
            param_to_index = {p.name: i for i, p in enumerate(schema.parameters)}
            for assignment in assignments:
                objects = [None] * len(schema.parameters)
                for var, val in assignment.items():
                    idx = param_to_index[f"?{var.lower()}"]
                    obj = self._name_to_object[val]
                    objects[idx] = obj
                ret_action = Action.new(self._problem, schema, objects)
                ret.append(ret_action)
        return ret

    def query(self, query: BaseRelation):
        return self._engine.query(query)

    def _init_template(self):
        self._template = Template()
        self._add_predicate_copies()
        self._add_object_information()
        self._add_derived_predicates()
        self._add_policy_rules()

        # add a derived predicate containing just the preconditions
        for schema in self._schemata:
            schema_name = schema.name
            head = self.relation_from_schema(schema_name, name=f"applicable_{schema_name}")
            body = self.get_schema_preconditions(schema_name)
            self._template += head <= body

    def print_state(self, state: list[Atom]):
        # may be extended and replaced
        pass

    @abstractmethod
    def _add_policy_rules(self):
        raise NotImplementedError

    @abstractmethod
    def _add_derived_predicates(self):
        raise NotImplementedError

    def _debug_inference(self):
        pass

    def _debug_inference_helper(self, relation: BaseRelation):
        print("-" * 80)
        rel_repr = str(relation).split("(")[0]
        print(rel_repr)
        # print("-" * len(rel_repr))
        results = self._engine.query(relation)
        relation_str = str(relation)
        results_repr = []
        for result in results:
            result_repr = "" + relation_str[:-1]
            for k, v in result.items():
                result_repr = result_repr.replace(k, v)
            results_repr.append(result_repr)
        results_repr = sorted(results_repr)
        print(" ".join(results_repr))

    def _debug_inference_actions(self):
        for schema in self._schemata:
            relation = self.relation_from_schema(schema)
            self._debug_inference_helper(relation)

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

    def add_rule(self, head_or_schema_name: Union[BaseRelation, str], body: list[BaseRelation]):
        assert isinstance(body, list)
        if isinstance(head_or_schema_name, BaseRelation):
            head = head_or_schema_name
            body = body
        else:
            assert isinstance(head_or_schema_name, str)
            schema_name = head_or_schema_name
            head = self.relation_from_schema(schema_name)
            body = body
            body += [self.relation_from_schema(schema_name, name=f"applicable_{schema_name}")]
        self._template += head <= body

    def relation_from_schema(self, schema: Schema, name=None) -> BaseRelation:
        """construct a relation object from a schema"""
        if isinstance(schema, str):
            schema = self._name_to_schema[schema]
        parameters = [p.name.replace("?", "").upper() for p in schema.parameters]
        parameters = [V.get(p) for p in parameters]
        if name is None:
            name = schema.name
        head = R.get(name)(parameters)
        return head

    def get_schema_preconditions(self, schema: Schema) -> list[BaseRelation]:
        """construct base body of a schema from its preconditions with typing"""
        if isinstance(schema, str):
            schema = self._name_to_schema[schema]
        schema: ActionSchema = schema

        body = []
        param_remap = {p.name: p.name.replace("?", "").upper() for p in schema.parameters}

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
            literal = R.get(predicate)(prec_vars)
            if p.negated:
                literal = ~literal
            body.append(literal)

        return body

    def get_ilg_facts(self, state: list[Atom]) -> list[StrAtom]:
        ret = []

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
                fact = StrAtom(predicate, [obj.name for obj in objects])
                ret.append(fact)

        return ret
