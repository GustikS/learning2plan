import os
import time
import warnings
from typing import Iterable, Union

import numpy as np
from neuralogic.core import R, Rule, Template, V, Transformation, Aggregation
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.dataset import FileDataset
from neuralogic.inference import EvaluationInferenceEngine
from neuralogic.nn.init import Initializer, Uniform
from neuralogic.nn.java import NeuraLogic
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.optim import SGD, Adam
from neuralogic.optim.optimizer import Optimizer
from neuralogic.optim.lr_scheduler import ArithmeticLR, GeometricLR
from pymimir import Action, Domain
from sklearn.metrics import f1_score
from typing_extensions import override

from modelling.templates import (anonymous_predicates, gnn_message_passing, object2object_edges,
                                 object_info_aggregation)
from policy_rules.policy.policy import Policy
from policy_rules.util.template_settings import (load_model_weights, neuralogic_settings,
                                                 save_template_model)
from policy_rules.util.timer import TimerContextManager


class LearningPolicy(Policy):

    def __init__(self, domain: Domain, debug=0):
        super().__init__(domain, debug)

        self.action_header2query = {
            query.predicate.name: query for schema in self._schemata if (query := self.relation_from_schema(schema))
        }

    def init_template(
            self,
            init_model: NeuraLogic = None,
            dim=1,
            num_layers=-1,
            state_regression=False,
            action_regression=False,
            **kwargs,
    ):
        self.dim = dim  # the general dimensionality of embeddings assumed in this model
        self.num_layers = num_layers  # the number of embedding message-passing-like layers

        super().init_template(init_model, **kwargs)

        if self.num_layers > 0:
            self.add_message_passing(self._template)

        if state_regression:  # add also an output head for the regression target
            self.add_rule(R.distance[1,], R.get(f"h_{self.num_layers}")("X")[1, self.dim], embedding_layer=-1)

        if state_regression or action_regression:
            neuralogic_settings.error_function

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
            new_predicate = new_predicate[prefix: self.dim, 1]
        self.add_rule(og_predicate, new_predicate, embedding_layer=-1)  # forbid the embeddings at input!

    @override
    def get_object_type(self, object_type: str, var_name: str) -> BaseRelation:
        """Object typing also starts from the input scalar dimension"""
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
        """this can be used for precise/complete debugging of the (neural) inference for each state"""
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
                    f"{neuron.getClass().getSimpleName()} : {predicate}{subs} : {neuron.getRawState().getValue()}"
                )
        print("\n".join(sorted(neurons)))
        print("=" * 80)

    def _debug_inference_helper(self, relation: BaseRelation, newline=True):
        print("-" * 80)
        results_repr = []
        relation.terms = [str(term).upper() for term in relation.terms]
        atom_values = self._engine.query(relation)
        if atom_values:
            try:
                for a in atom_values:
                    results_repr.append(f"{relation.predicate.name} {a[1]} : {a[0]}")
            except IndexError:
                results_repr = [relation.predicate.name + " <- true"]
        else:
            results_repr = [relation.predicate.name + " <- no inference"]

        if newline:
            print("\n".join(results_repr))
        else:
            print(" ".join(results_repr))

    @override
    def add_rule(
            self,
            head_or_schema_name: Union[BaseRelation, str],
            body: list[BaseRelation],
            guard_level: int = -1,  # = only call the rule after N inference steps
            embedding_layer=None,
            fixed_weight: Union[float, np.ndarray] = None,
    ):
        """Extending a given rule with weights and embeddings"""

        rule: Rule = self.get_rule(body, head_or_schema_name, guard_level=guard_level)
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
                rule.body += [self.add_weight(R.get(f"h_{embedding_layer}")(var), dim) for var in variables]
            else:
                rule.body.append(self.add_weight(R.get(f"h_{embedding_layer}"), dim))
        self._template += rule

    def add_weight(self, literal: BaseRelation, dim: int) -> WeightedRelation:
        if dim <= 0:
            return literal

        if not isinstance(literal, WeightedRelation) and not literal.negated:  # not yet weighted
            # if literal.predicate.name.startswith('applicable_'):
            #     return literal[dim, dim]
            if literal.predicate.name[:3] in ["ap_", "ag_", "ug_"]:  # scalar input atoms
                return literal[dim, 1]
            if literal.predicate.name in self.action_header2query.keys():  # scalar output actions
                return literal[dim, 1]
            if literal.predicate.name.startswith("g_") and literal.predicate.arity == 0:  # scalar special guards
                return literal[dim, 1]
            if literal.predicate.name == "satellite":
                # DZC 15/07/2024: I ended up using your nullary code, so not sure if this is needed anymore
                return literal[dim, 1]  # a hot patch for Dillon's dummy satellite(S) predicate
            else:
                return literal[dim, dim]
        else:
            if self._debug > 4:
                print(f"attempting to weight atom {literal} that is already weighted")
            return literal

    def add_message_passing(self, template: Template):
        """This is where we can incorporate various generic modelling/ML constructs to the template"""
        predicates = {pred.name: pred.arity for pred in self._domain.predicates}
        dim = self.dim if self.dim > 0 else 1

        template += anonymous_predicates(predicates, dim, input_dim=dim)

        # In the learning inference MAP ALSO THE UNACHIEVED PREDICATES!
        ug_predicates = {f"ug_{pred}": arity for pred, arity in predicates.items()}
        template += anonymous_predicates(ug_predicates, dim, input_dim=1)

        template += object_info_aggregation(max(predicates.values()), dim)
        # template += atom_info_aggregation(max(predicates.values()), dim)

        if self.num_layers > 1:  # start with some message-passing
            template += object2object_edges(max(predicates.values()), dim, "edge")

            # template += custom_message_passing("edge", "h0", dim)
            template += gnn_message_passing("edge", dim, num_layers=self.num_layers - 1)
            # template += gnn_message_passing(f"{2}-ary", dim, num_layers=num_layers)

    def train_model_from(
            self,
            train_data_dir: str,
            samples_limit: int = -1,
            weight_init: Initializer = Uniform(),
            num_epochs: int = 100,
            optimizer: Optimizer = Adam,
            learning_rate: float = 0.001,  # increase for SGD
            learning_rate_decay: Union["arithmetic", "geometric"] = "arithmetic",
            activations: Transformation = Transformation.LEAKY_RELU,
            aggregations: Aggregation = Aggregation.AVG,
            state_regression=False,
            action_regression=False,
            save_drawing=None,
    ):
        """Set up training, then call self._train_parameters for main training"""
        if state_regression or action_regression:
            neuralogic_settings.error_function = MSE()
        else:
            neuralogic_settings.error_function = CrossEntropy(with_logits=False)

        neuralogic_settings.initializer = weight_init

        match learning_rate_decay:
            case "arithmetic":
                decay = ArithmeticLR(num_epochs)
            case "geometric":
                quotient = 0.1  # these can be played with...
                every_n_steps = int(num_epochs / 10)
                decay = GeometricLR(quotient, every_n_steps)
            case "":
                decay = None
            case _:
                raise ValueError("Unrecognized learning rate decay method")

        neuralogic_settings.optimizer = optimizer(lr=learning_rate, lr_decay=decay)

        # these can be also set for specific rules in the template (keeping it global for simplicity)
        neuralogic_settings.rule_transformation = activations
        neuralogic_settings.relation_transformation = activations
        neuralogic_settings.rule_aggregation = aggregations
        # the output neuron activation should get set automatically w.r.t. given setting (classification/regression)

        with TimerContextManager("building template"):
            self.model = self._template.build(neuralogic_settings)

        if save_drawing is not None:
            self.model.draw(filename=save_drawing)
            print("Saved template visualisation to", save_drawing)

        self._engine.model = self.model

        if samples_limit > 0:
            neuralogic_settings["stratification"] = False  # skip this to keep the order of samples (for debugging)
            neuralogic_settings["appLimitSamples"] = samples_limit
            print(f"Starting building the samples with a limit to the first {samples_limit}")
        else:
            print(f"Starting building all samples")

        self._train_parameters(train_data_dir, epochs=num_epochs)

    def _train_parameters(self, lrnn_dataset_dir: str, epochs: int = 100):
        try:
            dataset = FileDataset(f"{lrnn_dataset_dir}/examples.txt", f"{lrnn_dataset_dir}/queries.txt")
            neural_samples = self.model.build_dataset(dataset)
            print("Neural samples successfully built (the template logic is working correctly)!")
            if self._debug > 1:
                self._debug_neural_samples(neural_samples)

            # Main training loop
            # TODO: maybe have a validation split to save best model
            # there is a whole range of options for that in the backend, with detailed reporting of the progress across various metrics,
            # but that will only work if the whole training is performed there (i.e. not in python epoch by epoch) and logging turned (to FINE at least)...
            # ...on the other hand it's more flexible to control it from python here, so let's continue with this I guess
            best_f1 = 0
            best_state_dict = self.model.state_dict()
            best_epoch = -1
            with TimerContextManager("training the LRNN"):
                for epoch in range(epochs):
                    t = time.time()
                    results, n_samples = self.model(neural_samples, True, epochs=1)
                    if decay := neuralogic_settings.optimizer._lr_decay:
                        decay.decay(epoch)  # if we go epoch by epoch this needs to be called manually
                    t = time.time() - t

                    # result[0] = target
                    # result[1] = prediction
                    # result[2] = difference between the prediction and target

                    y_true = np.array([result[0] for result in results])
                    y_pred = np.array([result[1] for result in results])
                    y_pred_rounded = y_pred >= 0.5
                    diff = np.array([abs(result[2]) for result in results])

                    loss = sum(diff) / len(diff)
                    accuracy = sum(y_true == y_pred_rounded) / len(results)
                    f1 = f1_score(y_true, y_pred_rounded)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_state_dict = self.model.state_dict()
                        best_epoch = epoch

                    print(f"{epoch=}, {n_samples=}, {loss=}, {accuracy=}, {f1=}, {t=}")

            print(f"Best model at epoch={best_epoch} with f1_score={best_f1}")
            self.model.load_state_dict(best_state_dict)
        except KeyboardInterrupt:
            print(f"Training stopped early due to keyboard interrupt!")

        # Evaluate again after training
        if self._debug > 1:
            results = self.model(neural_samples, train=False)  # evaluate again due to missing sample outputs
            for i in range(len(neural_samples)):
                result = results[i]
                sample = neural_samples.samples[i]
                print(f"target: {sample.target} <- predicted: {result} for sample: {sample.java_sample.query.ID}")

        print("-" * 80)

        # DZC 15/07/2024: This seems redundant as store_policy is called after training in run.py so I commented it out
        # if save_model_path:
        #     store_template_model(self.model, save_model_path)

    def _debug_neural_samples(self, neural_samples):
        """Check that there are no problems and show some statistics"""
        state2actions = {}
        for sample in neural_samples:
            state_net = sample.java_sample.query.evidence.getId()  # a shared neural net for the given State
            actions = state2actions.get(state_net, [])
            actions.append((sample, sample.target, sample.java_sample.query.neuron))
            state2actions[state_net] = actions
        num_reachable_negative = 0
        num_multiple = 0
        for state, actions in state2actions.items():
            reachable = [action for action in actions if action[2]]
            if not reachable:
                raise Exception(f"State {state.getId()} has no reachable actions at all!")
            if len(reachable) > 1:
                num_multiple += 1
            reachable_positive = [action for action in reachable if action[1]]
            if not reachable_positive:
                raise Exception(f"State {state.getId()} has no reachable positive (optimal) actions!")
            reachable_negative = [action for action in reachable if not action[1]]
            if reachable_negative:
                num_reachable_negative += 1
        print(f"There are {len(neural_samples)} learning queries across {len(state2actions)} unique states")
        print(f"{float(num_multiple) / len(state2actions) * 100} % of states have more than 1 action derived")
        print(
            f"Only {float(num_reachable_negative) / len(state2actions) * 100} % of states "
            f"have some negative action derived, and hence can be improved with learning"
        )

    def reset_parameters(self):
        self.model.reset_parameters()

    def load_parameters(self, weights_file_path: str = None):
        load_model_weights(self.model, weights_file_path)

    def store_policy(self, save_path: str):
        save_template_model(self.model, save_path)


class FasterLearningPolicy(LearningPolicy):
    """Experimental speedup version by avoiding all duplicit and redundant computation of the inference engine"""

    def __init__(self, domain: Domain, debug=0):
        super().__init__(domain, debug)

    def init_template(self, init_model: NeuraLogic = None, dim=1, num_layers=-1, **kwargs):
        super().init_template(init_model, dim=dim, num_layers=num_layers, **kwargs)
        # self.model.settings['neuralNetsPostProcessing'] = False  # for speedup
        # self.model.settings.chain_pruning = False     # if trained with pruning we should keep it for evaluation too
        self.model.settings.iso_value_compression = False

    @override
    def query_actions(self) -> list[(float, Action)]:
        try:
            self._engine.dataset.samples[0].query = None  # remove any query if present - we ask all queries at once!
            built_dataset = self.model.build_dataset(self._engine.dataset)
            self._built_state_network = built_dataset[0]
            # we cannot skip this extra evaluation if we want alignment with the evaluation_inference_engine
            output = self.model(built_dataset.samples, train=False)
        except Exception as e:
            warnings.warn(f"Failed to build template on the state: {self._engine.dataset}")
            warnings.warn(str(e))
            raise e
        return super().query_actions()

    @override
    def get_action_substitutions(self, action_name: str) -> tuple[float, dict]:
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
