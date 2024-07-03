import os
import warnings
from typing import Union, Iterable
from typing_extensions import override

import numpy as np
from neuralogic.core import R, Rule, V, Template
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.inference import EvaluationInferenceEngine
from neuralogic.dataset import FileDataset
from neuralogic.nn.java import NeuraLogic
from pymimir import Domain, Action

from modelling.templates import anonymous_predicates, object_info_aggregation, object2object_edges, gnn_message_passing
from policy_rules.policy.policy import Policy
from policy_rules.util.template_settings import neuralogic_settings, store_template_model, load_model_weights


class LearningPolicy(Policy):

    def __init__(self, domain: Domain, debug=0):
        super().__init__(domain, debug)

        self.action_header2query = {query.predicate.name: query
                                    for schema in self._schemata
                                    if (query := self.relation_from_schema(schema))}

    def init_template(self, init_model: NeuraLogic = None, dim=1, num_layers=-1):
        self.dim = dim  # the general dimensionality of embeddings assumed in this model
        self.num_layers = num_layers  # the number of embedding message-passing-like layers

        super().init_template(init_model)

        if self.num_layers > 0:
            self.add_message_passing(self._template)

        # An inference engine just like in Policy, but this one returns the numeric values also
        self._engine = EvaluationInferenceEngine(self._template, neuralogic_settings)

        if init_model:
            self.model = init_model
            self._engine.model = init_model
        else:
            self.model = self._engine.model

    @override
    def get_action_substitutions(self, action_name: str) -> Iterable[tuple[float, dict]]:
        action_query = self.action_header2query[action_name]
        # there is a little bug/discrepancy between the eval/inference in neuralogic
        # so we need to reformat a bit here (upper string instead of capital var) - will correct that in next version
        for term in action_query.terms:
            term.name = term.name.upper()
        substitutions = list(self._engine.query(action_query))
        assignments = []
        for val, subs in substitutions:
            subss = {}
            for var, const in subs.items():
                subss[var.name.capitalize()] = const
            assignments.append((val, subss))
        return assignments

    def add_input_predicate(self, og_predicate, new_predicate):
        """The input predicate mapping from scalar 1 to a given embedding dimension vector"""
        prefix = new_predicate.predicate.name[:2]
        og_predicate = og_predicate
        if self.dim > 0:
            new_predicate = new_predicate[prefix:self.dim, 1]
        self.add_rule(og_predicate, new_predicate, embedding_layer=-1)  # forbid the embeddings at input!

    @override
    def get_object_type(self, object_type: str, var_name: str) -> BaseRelation:
        type = R.get(object_type)(V.get(var_name))
        if self.dim > 0:
            type = type[self.dim, 1]
        return type

    def add_output_action(self, head, body):
        """The output action predicate mapping from the given embedding dimension back to a scalar value"""
        rule = self.get_rule(body, head)
        if self.dim > 0:
            head = rule.head[1, self.dim]
        else:
            head = rule.head
        self.add_rule(head, rule.body)

    def _debug_template(self):
        print("=" * 80)
        print("Template:")
        print(self._template)
        print("=" * 80)

    def _debug_inference(self):
        super()._debug_inference()
        print("=" * 80)
        print("Debugging all state neuron values:")
        self._engine.dataset[0].query = None
        built_dataset = self.model.build_dataset(self._engine.dataset)
        atom_values = built_dataset[0]._get_literals()
        neurons = []
        for predicate, substitutions in atom_values.items():
            for subs, neuron in substitutions.items():
                neurons.append(
                    f'{neuron.getClass().getSimpleName()} : {predicate}{subs} : {neuron.getRawState().getValue()}')
        print('\n'.join(sorted(neurons)))
        print("=" * 80)

    def _debug_inference_helper(self, relation: BaseRelation, newline=True):
        print("-" * 80)
        results_repr = []
        relation.terms = [str(term).upper() for term in relation.terms]
        atom_values = self._engine.query(relation)
        if atom_values:
            for a in atom_values:
                results_repr.append(f'{relation.predicate.name} {a[1]} : {a[0]}')
        else:
            results_repr = [relation.predicate.name + " <- no inference"]

        if newline:
            print("\n".join(results_repr))
        else:
            print(" ".join(results_repr))

    @override
    def add_rule(self,
                 head_or_schema_name: Union[BaseRelation, str],
                 body: list[BaseRelation],
                 embedding_layer=None,
                 fixed_weight: Union[float, np.ndarray] = None):

        rule: Rule = self.get_rule(body, head_or_schema_name)
        dim = self.dim

        if fixed_weight:  # assign a given fixed weight
            if not isinstance(rule.head, WeightedRelation):
                rule.head = rule.head[fixed_weight]
            rule.head.fixed()
        elif dim > 0:  # we want to add weights
            rule.head = self.add_weight(rule.head, dim)
            for i in range(len(rule.body)):
                rule.body[i] = self.add_weight(rule.body[i], dim)

        if not embedding_layer:
            embedding_layer = self.num_layers

        if embedding_layer > 0:  # add object embeddings
            variables = set(rule.head.terms)
            for lit in rule.body:
                variables.update(lit.terms)
            if variables:
                rule.body += [self.add_weight(R.get(f'h_{embedding_layer}')(var), dim) for var in variables]
            else:
                rule.body.append(self.add_weight(R.get(f'h_{embedding_layer}'), dim))
        self._template += rule

    def add_weight(self, literal: BaseRelation, dim: int) -> WeightedRelation:
        if dim <= 0:
            return literal

        if not isinstance(literal, WeightedRelation) and not literal.negated:  # not yet weighted
            # if literal.predicate.name.startswith('applicable_'):
            #     return literal[dim, dim]
            if literal.predicate.name[:3] in ['ap_', 'ag_', 'ug_']:  # scalar inputs
                return literal[dim, 1]
            else:
                return literal[dim, dim]
        else:
            if self._debug > 3:
                print(f"{literal} is already weighted")
            return literal

    def add_message_passing(self, template: Template):
        """This is where we can incorporate various generic modelling/ML constructs to the template"""
        predicates = {pred.name: pred.arity for pred in self._domain.predicates}
        dim = self.dim if self.dim > 0 else 1

        template += anonymous_predicates(predicates, dim, input_dim=dim)

        # In the learning inference MAP ALSO THE UNACHIEVED PREDICATES!
        ug_predicates = {f'ug_{pred}': arity for pred, arity in predicates.items()}
        template += anonymous_predicates(ug_predicates, dim, input_dim=1)

        template += object_info_aggregation(max(predicates.values()), dim)
        # template += atom_info_aggregation(max(predicates.values()), dim)

        template += object2object_edges(max(predicates.values()), dim, "edge")

        # template += custom_message_passing("edge", "h0", dim)
        template += gnn_message_passing("edge", dim, num_layers=self.num_layers)
        # template += gnn_message_passing(f"{2}-ary", dim, num_layers=num_layers)

    def train_model_from(self, train_data_dir: str):
        if train_data_dir.endswith("/_"):
            train_data_dir = train_data_dir[:-2]
        assert os.path.isdir(train_data_dir), print(f"No LRNN training data available at {train_data_dir}")
        try:
            self.train_parameters(train_data_dir)
        except Exception as e:
            print(f"Invalid training setup from: {train_data_dir}")
            print(e)

    def train_parameters(self, lrnn_dataset_dir: str, epochs: int = 100, save_model_path: str = None):
        try:
            dataset = FileDataset(f"{lrnn_dataset_dir}/examples.txt", f"{lrnn_dataset_dir}/queries.txt")
            neural_samples = self.model.build_dataset(dataset)
        except Exception as e:
            print(f"An error occured during attempt to train policy model from: {lrnn_dataset_dir}")
            print(e)
        results = self.model(neural_samples, train=True, epochs=epochs)
        print(results)
        if save_model_path:
            store_template_model(self.model, save_model_path)

    def reset_parameters(self):
        self.model.reset_parameters()

    def load_parameters(self, weights_file_path: str = None):
        load_model_weights(self.model, weights_file_path)

    def store_policy(self, save_path: str):
        store_template_model(self.model, save_path)


class FasterLearningPolicy(LearningPolicy):
    """Experimental speedup version by avoiding all duplicit and redundant computation"""

    def __init__(self, domain: Domain, debug=0):
        super().__init__(domain, debug)

    def init_template(self, init_model: NeuraLogic = None, dim=1, num_layers=-1):
        super().init_template(init_model, dim=dim, num_layers=num_layers)
        # self.model.settings['neuralNetsPostProcessing'] = False  # for speedup
        # self.model.settings.chain_pruning = False     # if trained with pruning we should keep it for evaluation too
        self.model.settings.iso_value_compression = False

    @override
    def query_actions(self) -> list[(float, Action)]:
        try:
            built_dataset = self.model.build_dataset(self._engine.dataset)
            self._built_state_network = built_dataset[0]
            # we cannot skip this extra evaluation if we want alignment with the evaluation_inference_engine
            output = self.model(built_dataset.samples, train=False)
        except Exception:
            warnings.warn(f"Failed to build template on the state: {self._engine.dataset}")
        return super().query_actions()

    @override
    def get_action_substitutions(self, action_name: str) -> (float, dict):
        atoms = self._built_state_network.get_atom(self.action_header2query[action_name])
        if atoms:
            result = [(a.value, a.substitutions) for a in atoms]
        else:
            result = []

        if self._debug > 2:
            check_result = super().get_action_substitutions(action_name)
            check = str(check_result).lower()
            fast = str(result).lower()
            if fast != check:
                print("-" * 80)
                print("Debugging error: mismatch between inference engine and fast evaluation")
                print(fast)
                print(check)
                # raise RuntimeError(f"Results mismatch between standard and fast policy evaluation")

        return result
