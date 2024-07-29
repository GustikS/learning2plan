"""Make use of neuralogic and pymimir to encode policies as Horn clause rules"""

from abc import abstractmethod
from itertools import product
from typing import Iterable, Union

from neuralogic.core import C, R, Template, V
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.inference.inference_engine import InferenceEngine
from neuralogic.nn.java import NeuraLogic
from pymimir import Action, ActionSchema, Atom, Domain, Literal, Object, Problem

from policy_rules.util.str_atom import StrAtom
from policy_rules.util.template_settings import neuralogic_settings, save_template_model

Schema = Union[str, ActionSchema]


class Policy:
    def __init__(self, domain: Domain, debug=0):
        self._domain = domain
        self._debug = debug

        self._schemata = self._domain.action_schemas
        self._predicates = self._domain.predicates
        self._name_to_schema: dict[str, ActionSchema] = {schema.name: schema for schema in self._schemata}
        self._prev_state = None

        self.guards_levels = -1

        self._statics: set[str] = set()  # to define manually for state print debugging

    def init_template(self, init_model: NeuraLogic = None, **kwargs):
        if init_model:
            template = Template()
            template.template = init_model.source_template
            self._template = template
        else:
            self._init_template(**kwargs)

        self._engine = InferenceEngine(self._template, neuralogic_settings)

    def _init_template(self, include_knowledge=True, **kwargs):
        self._template = Template()
        self._add_predicate_copies()

        if include_knowledge:
            try:
                self._add_derived_predicates()
                self._add_policy_rules()
            except NotImplementedError as e:
                print("Domain knowledge is missing for this domain, will resort to generic policy learning.")
                print(e)
            except Exception as e:
                print(e)
            prefix = "applicable_"
        else:
            prefix = ""

        # add a derived predicate containing just the preconditions
        for schema in self._schemata:
            schema_name = schema.name
            head = self.relation_from_schema(schema_name, name=f"{prefix}{schema_name}")
            body = self.get_schema_preconditions(schema_name, **kwargs)
            if include_knowledge:
                self.add_rule(head, body)
            else:
                self.add_output_action(head, body)

        if self.guards_levels > -1:
            self._template += R.get("g_0")  # if we want to use the inference hierarchy, starting in the sample...

    def setup_test_problem(self, problem: Problem):
        """Set up a STATEFUL dependency on a current test problem"""
        # todo remove this stateful dependence completely and just pass it as an argument (after asking Dillon) ?
        # DZC 27/06/24: The reason why this may be useful is if we want to use the same policy for
        # different problems in the same domain. Although it is probably more robust to just
        # reinstantiate for each problem like you mentioned. However, since it works, I'll just
        # keep the code here for now.
        # GS 27/06/24: but that is exactly why it is not useful, no? If the same policy should work
        # for different problems, then there should be nothing problem-specific stored in it, no?
        # Completely agree with storing for a given domain though.
        self._problem = problem
        self._objects = self._problem.objects
        self._goal = self._problem.goal
        self._name_to_object: dict[str, Object] = {obj.name: obj for obj in self._objects}

    def _get_atoms_from_state(self, state: list[Atom]):
        ilg_atoms = self.get_ilg_facts(state)
        lrnn_atoms = [R.get(atom.predicate)([C.get(obj) for obj in atom.objects]) for atom in ilg_atoms]
        object_atoms = self.get_object_information()
        return lrnn_atoms + object_atoms

    def setup_test_state(self, state: list[Atom]):
        """Set up a STATEFUL dependency on a current test State"""
        atoms = self._get_atoms_from_state(state)
        self._engine.set_knowledge(atoms)

    def solve(self, state: list[Atom]) -> list[(float, Action)]:
        """given a State from the currently assumed Problem, return possible actions from policy rules"""
        self.setup_test_state(state)

        if self._debug > 3:
            self._debug_template()

        if self._debug > 2:
            self._debug_inference()

        return self.query_actions()

    def query_actions(self) -> list[(float, Action)]:
        ret = []

        # single actions
        for schema in self._schemata:
            assignments = self.get_action_substitutions(schema.name)
            param_to_index = {p.name: i for i, p in enumerate(schema.parameters)}
            for value, assignment in assignments:
                objects = [None] * len(schema.parameters)
                for var, val in assignment.items():
                    idx = param_to_index[f"?{var.lower()}"]
                    obj = self._name_to_object[val]
                    objects[idx] = obj
                ret_action = Action.new(self._problem, schema, objects)
                ret.append((value, ret_action))

        return ret

    def get_action_substitutions(self, action_name: str) -> Iterable[tuple[float, dict]]:
        action_header = self.relation_from_schema(action_name)
        assignments = self._engine.query(action_header)
        for assignment in assignments:
            yield 1, assignment  # all action equally good here

    def print_state(self, state: list[Atom]):
        # may be extended and replaced
        statics = []
        nonstatics = []
        for atom in state:
            atom = atom.get_name()
            if atom.split("(")[0] in self._statics:
                statics.append(atom)
            else:
                nonstatics.append(atom)

        nonstaticsset = set(nonstatics)
        print()
        print("Statics:")
        for f in sorted(statics):
            print(f)

        print()
        print("Goal:")
        goals = []
        for g in self._goal:
            assert not g.negated
            goals.append(g.atom.get_name())
        for g in sorted(goals):
            if g in nonstaticsset:
                g += " *"
            else:
                g += " x"
            print(g)

        print()
        print("Current state:")
        for f in sorted(nonstatics):
            print(f)

        self._prev_state = state

    @abstractmethod
    def _add_policy_rules(self):
        raise NotImplementedError

    @abstractmethod
    def _add_derived_predicates(self):
        raise NotImplementedError

    @abstractmethod
    def _debug_template(self, serialise=None):
        pass
        # print("=" * 80)
        # print("Template for current state:")
        # print(self._template)
        # # print("=" * 80)
        # print("-" * 80)
        # for a in sorted(self._engine.examples, key=lambda x: str(x)):
        #     print(a)
        # print("=" * 80)
        # # print(self._engine.examples)

    def _debug_inference(self):
        pass

    def _debug_inference_helper(self, relation: BaseRelation, newline=False):
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
        if newline:
            print("\n".join(results_repr))
        else:
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
            # the only place with a hardcoded weight - a shared scalar based on the prefix
            new_predicate = R.get(f"{prefix}_{predicate.name}")(variables)
            og_predicate = R.get(predicate.name)(variables)
            self.add_input_predicate(og_predicate, new_predicate)

    def add_guard_hierarchy(self, to: int):
        if self.guards_levels < 0:
            # self._template += R.get(f'h_{0}')() <= R.special.true
            self.guards_levels = 0
        for i in range(self.guards_levels, to):
            self._template += R.get(f"g_{i + 1}")() <= R.get(f"g_{i}")()
        self.guards_levels = to

    def get_guard_atom(self, guard_level):
        if self.guards_levels < guard_level:
            self.add_guard_hierarchy(guard_level)
        if guard_level > 0:
            return R.get(f"g_{guard_level}")

    def add_input_predicate(self, og_predicate, new_predicate):
        self.add_rule(og_predicate, new_predicate)

    def add_output_action(self, head, body):
        self.add_rule(head, body)

    def get_object_information(self) -> list[BaseRelation]:
        """return object type facts"""
        object_types = []
        for obj in self._problem.objects:
            assert obj.is_constant()
            # TODO intermediate types?  - you can e.g. add rules supertype(X) :- subtype(X). for each type
            # DZC 27/06/24: That is true, I'm too lazy to do that now and everything seems to work for the current domains
            object_types.append(R.get(obj.type.name)(C.get(obj.name)))
            # self._template += R.get(obj.type.base.name)(C.get(obj.name))
        return object_types

    def add_rule(self, head_or_schema_name: Union[BaseRelation, str], body: list[BaseRelation], **kwargs):
        self._template += self.get_rule(body, head_or_schema_name, **kwargs)

    def get_rule(
        self,
        body: Union[list | BaseRelation],
        head_or_schema_name: Union[str | BaseRelation],
        guard_level: int = -1,
        **kwargs,
    ):
        assert isinstance(body, Union[list | BaseRelation])
        if isinstance(head_or_schema_name, BaseRelation):
            head = head_or_schema_name
            body = body
        else:
            assert isinstance(head_or_schema_name, str)
            schema_name = head_or_schema_name
            head = self.relation_from_schema(schema_name)
            body = body
            body += [self.relation_from_schema(schema_name, name=f"applicable_{schema_name}")]
        if guard_level > -1:
            atom = self.get_guard_atom(guard_level)
            if atom:
                if not isinstance(body, list):
                    body = [body]
                body.append(atom)
        return head <= body

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

    def get_schema_preconditions(self, schema: Schema, add_types=True, **kwargs) -> list[BaseRelation]:
        """construct base body of a schema from its preconditions with typing"""
        if isinstance(schema, str):
            schema = self._name_to_schema[schema]
        schema: ActionSchema = schema

        body = []
        param_remap = {p.name: p.name.replace("?", "").upper() for p in schema.parameters}

        if add_types:
            ## add variables and their types
            for param in schema.parameters:
                assert param.is_variable()
                remap = param_remap[param.name]
                object_type = param.type.name
                atom = self.get_object_type(object_type, remap)
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

    def get_object_type(self, object_type: str, var_name: str):
        return R.get(object_type)(V.get(var_name))

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

    def store_policy(self, save_path: str):
        save_template_model(self._template, save_path)
