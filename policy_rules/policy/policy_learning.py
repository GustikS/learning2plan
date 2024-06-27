import os
import pickle
import warnings
from typing import Union, Iterable

from neuralogic.dataset import Dataset, Sample
from typing_extensions import override

from neuralogic.core import Template, Settings, R, C, Transformation, Aggregation, Rule
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.inference import EvaluationInferenceEngine
from pymimir import Domain, Problem, Atom, Action

from modelling.samples import get_filename
from modelling.testing import load_model
from modelling.training import build_samples, store_weights, store_template
from policy_rules.policy.policy import Policy, Schema


class LearningPolicy(Policy):

    def __init__(self, domain: Domain, template_path: str, debug=0, train: bool = True):
        # TODO remove the Problem dependency - a policy should generalize over different problems, right?
        super().__init__(domain, template_path, debug, train)

        # we can setup all the learning/numeric-evaluation-related settings here
        self.settings = Settings(
            iso_value_compression=False,
            chain_pruning=True,
            rule_transformation=Transformation.TANH,  # change to RELU for better training
            rule_aggregation=Aggregation.SUM,  # change to avg for better generalization
            relation_transformation=Transformation.SIGMOID,  # change to RELU for better training - check label match
            epochs=100
        )

        self.setup_template(domain, template_path, train)

        # an inference engine just like in Policy, but this one returns the numeric values also
        self._engine = EvaluationInferenceEngine(self._template, self.settings)
        self.model = self._engine.model

        self.action_header2query = {query.predicate.name: query
                                    for schema in self._schemata
                                    if (query := self.relation_from_schema(schema))}

    def setup_template(self, domain: Domain, template_path: str, train: bool):
        """Initialize a learning policy template - either load from file or create a new one and possibly train + store"""
        try:
            _, self._template = load_model(template_path)
        except FileNotFoundError:
            print(f"No stored template found at {template_path} - will train a default one and store it there instead!")
            # self._init_template()     - called in superclass already
            if train:
                training_data_path = get_filename(domain.name, False, 'lrnn', "..", "")
                if os.path.isdir(training_data_path):
                    try:
                        LearningPolicy.train_parameters(self._template.build(self.settings),
                                                        training_data_path,
                                                        save_model_path=template_path)
                    except Exception as e:
                        print(f"Invalid training setup from: {training_data_path}")
                        print(e)
                else:
                    print(f"No LRNN-format training data available at {training_data_path}")
            else:
                print("Training not allowed - resorting to a pure handcrafted template")

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

    @staticmethod
    def train_parameters(model, lrnn_dataset_path: str, epochs: int = 100, save_model_path: str = None):
        neural_samples, logic_samples = build_samples(model, lrnn_dataset_path)
        results = model(neural_samples, train=True, epochs=epochs)
        print(results)
        if save_model_path:
            store_weights(model, save_model_path)
            store_template(model, save_model_path)

    def reset_parameters(self):
        self.model.reset_parameters()

    def load_parameters(self, file_path: str = None):
        with open(file_path + "_weights", 'rb') as f:
            weights = pickle.load(f)
            self.model.load_state_dict(weights)


class FasterEvaluationPolicy(LearningPolicy):
    """Experimental speedup version by avoiding all duplicit and redundant computation"""

    def __init__(self, domain: Domain, template_path: str, debug=0, train: bool = True):
        super().__init__(domain, template_path, debug, train)

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
