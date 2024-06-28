import os
import pickle
import warnings
from typing import Union, Iterable

from neuralogic.core import R, Rule
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.inference import EvaluationInferenceEngine
from neuralogic.nn.java import NeuraLogic
from pymimir import Domain, Action
from typing_extensions import override

from modelling.training import build_samples
from policy_rules.policy.policy import Policy
from policy_rules.util.template_settings import neuralogic_settings, store_template_model, load_model_weights


class LearningPolicy(Policy):

    def __init__(self, domain: Domain, init_model: NeuraLogic = None, debug=0):
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

    @override
    def add_rule(self,
                 head_or_schema_name: Union[BaseRelation, str],
                 body: list[BaseRelation],
                 dim: int = 1,
                 embedding_layer: int = -1):
        rule: Rule = self.get_rule(body, head_or_schema_name)
        if dim > 0:  # we want weights
            rule.head = self.add_weight(rule.head, dim)
            for i in range(len(rule.body)):
                rule.body[i] = self.add_weight(rule.body[i], dim)
        if embedding_layer > 0:  # add object embeddings
            variables = rule.head.terms
            for lit in rule.body:
                variables += lit.terms
            rule.body += [R.get(f'h_{embedding_layer}')(var)[dim, dim] for var in variables]
        self._template += rule

    def add_weight(self, literal: BaseRelation, dim: int) -> WeightedRelation:
        if not isinstance(literal, WeightedRelation) and not literal.negated:  # not yet weighted
            return literal[dim, dim]
        else:
            if self._debug > 2:
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
