import os
import pickle
import warnings
from typing import Union, Iterable

import numpy as np
from neuralogic.core import R, Rule, V
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.inference import EvaluationInferenceEngine
from neuralogic.nn.java import NeuraLogic
from pymimir import Domain, Action
from typing_extensions import override

from modelling.training import build_samples
from policy_rules.policy.policy import Policy
from policy_rules.util.template_settings import neuralogic_settings, store_template_model, load_model_weights


class LearningPolicy(Policy):

    def __init__(self, domain: Domain, init_model: NeuraLogic = None, debug=0, dim=1, num_layers=1):
        self.dim = dim
        self.num_layers = num_layers

        super().__init__(domain, init_model, debug)

        # An inference engine just like in Policy, but this one returns the numeric values also
        self._engine = EvaluationInferenceEngine(self._template, neuralogic_settings)

        if init_model:
            self.model = init_model
            self._engine.model = init_model
        else:
            self.model = self._engine.model

        self.action_header2query = {query.predicate.name: query
                                    for schema in self._schemata
                                    if (query := self.relation_from_schema(schema))}

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
        self.add_rule(og_predicate, new_predicate[prefix: self.dim, 1], embedding_layer=-1)

    @override
    def get_object_type(self, object_type: str, var_name: str) -> BaseRelation:
        return R.get(object_type)(V.get(var_name))[self.dim, 1]

    def add_output_action(self, head, body):
        """The output action predicate mapping from the given embedding dimension back to a scalar value"""
        rule = self.get_rule(body, head)
        self.add_rule(rule.head[1, self.dim], rule.body)

    def _debug_template(self):
        print("=" * 80)
        print("Template:")
        print(self._template)
        print("=" * 80)

    def _debug_inference(self):
        super()._debug_inference()
        print("=" * 80)
        print("All state neuron values:")
        built_dataset = self.model.build_dataset(self._engine.dataset)
        atom_values = built_dataset[0]._get_literals()
        for predicate, substitutions in atom_values.items():
            for subs, neuron in substitutions.items():
                print(neuron.getClass().getSimpleName(), predicate, subs, ':', neuron.getRawState().getValue())
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
                 embedding_layer: int = -1,
                 fixed_weight: Union[float, np.ndarray] = None):

        rule: Rule = self.get_rule(body, head_or_schema_name)
        dim = self.dim

        if fixed_weight:  # assign a given fixed weight
            if not isinstance(rule.head, WeightedRelation):
                rule.head = rule.head[fixed_weight]
            rule.head.fixed()
        elif dim > 0:  # we want learnable weights
            rule.head = self.add_weight(rule.head, dim)
            for i in range(len(rule.body)):
                rule.body[i] = self.add_weight(rule.body[i], dim)
        if embedding_layer > 0:  # add object embeddings
            variables = rule.head.terms
            for lit in rule.body:
                variables += lit.terms
            if variables:
                rule.body += [R.get(f'h_{embedding_layer}')(var)[dim, dim] for var in variables]
            else:
                rule.body.append(R.get(f'h_{embedding_layer}')[dim, dim])
        self._template += rule

    def add_weight(self, literal: BaseRelation, dim: int) -> WeightedRelation:
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
            neural_samples, logic_samples = build_samples(self.model, lrnn_dataset_dir)
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


class FasterEvaluationPolicy(LearningPolicy):
    """Experimental speedup version by avoiding all duplicit and redundant computation"""

    def __init__(self, domain: Domain, init_model: NeuraLogic = None, debug=0):
        super().__init__(domain, init_model, debug)

        # self.model.settings['neuralNetsPostProcessing'] = False  # for speedup
        # self.model.settings.chain_pruning = False     # if trained with pruning we should keep it for evaluation too
        self.model.settings.iso_value_compression = False

    @override
    def query_actions(self) -> list[(float, Action)]:
        try:
            built_dataset = self.model.build_dataset(self._engine.dataset)
            self._built_state_network = built_dataset[0]
            output = self.model(built_dataset.samples, train=False)  # todo test if we can skip this
        except Exception:
            warnings.warn(f"Failed to build template on the state: {self._engine.dataset}")
        return super().query_actions()

    @override
    def get_action_substitutions(self, action_name: str) -> (float, dict):
        atoms = self._built_state_network.get_atom(self.action_header2query[action_name])
        if atoms:
            for a in atoms:
                yield a.value, a.substitutions
        else:
            pass
            # print(f"Failed to evaluate action{action_name} at state: {self._engine.dataset}")
