import os
import time
import warnings
from contextlib import contextmanager
from typing import Iterable, Union

import jpype
import numpy as np
from neuralogic.core import Aggregation, R, Rule, Template, Transformation, V
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.dataset import FileDataset
from neuralogic.inference import EvaluationInferenceEngine
from neuralogic.nn.init import Initializer, Uniform
from neuralogic.nn.java import NeuraLogic
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.optim import SGD, Adam
from neuralogic.optim.lr_scheduler import ArithmeticLR, GeometricLR
from neuralogic.optim.optimizer import Optimizer
from pymimir import Action, Domain
from sklearn.metrics import f1_score
from termcolor import colored
from tqdm import tqdm
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

        results_repr = sorted(results_repr, key=lambda x: str(x))

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
            square = True

            if literal.predicate.name[:3] in ["ap_", "ag_", "ug_"]:
                # scalar input atoms
                square = False
            if literal.predicate.name in self.action_header2query.keys():
                # scalar output actions
                square = False
            if literal.predicate.name.startswith("g_") and literal.predicate.arity == 0:
                # scalar special guards
                square = False

            if square:
                ret = literal[dim, dim]
            else:
                ret = literal[dim, 1]

            return ret
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
            learning_rate: float = 0.0001,  # increase for SGD
            learning_rate_decay: str = "",
            activations: Transformation = Transformation.TANH,
            aggregations: str = "max",
            state_regression=False,
            action_regression=False,
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
                raise ValueError("Unrecognized learning rate decay method.")

        neuralogic_settings.optimizer = optimizer(lr=learning_rate, lr_decay=decay)

        match aggregations:
            case "max":
                aggregations = Aggregation.MAX
            case "sum":
                aggregations = Aggregation.SUM
            case "mean":
                aggregations = Aggregation.AVG

        # these can be also set for specific rules in the template (keeping it global for simplicity)
        neuralogic_settings.rule_transformation = activations
        neuralogic_settings.relation_transformation = activations
        neuralogic_settings.rule_aggregation = aggregations
        # the output neuron activation should get set automatically w.r.t. given setting (classification/regression)

        if activations == Transformation.RELU or activations == Transformation.LEAKY_RELU:
            neuralogic_settings.iso_value_compression = False  # these are not compatible with the speedup via lifting

        with TimerContextManager("building template"):
            self.model = self._template.build(neuralogic_settings)

        self._engine.model = self.model

        if samples_limit > 0:
            neuralogic_settings["stratification"] = False  # skip this to keep the order of samples (for debugging)
            neuralogic_settings["appLimitSamples"] = samples_limit
            print(f"Starting building the samples with a limit to the first {samples_limit}")
        else:
            print(f"Starting building all samples")

        if self._debug > 2:
            self._grounding_debug()

        self._train_parameters(train_data_dir, epochs=num_epochs)

    def _train_parameters(self, lrnn_dataset_dir: str, epochs: int = 100):
        try:
            dataset = FileDataset(f"{lrnn_dataset_dir}/examples.txt", f"{lrnn_dataset_dir}/queries.txt")
            progress = self._grounding_progress()
            neural_samples = self.model.build_dataset(dataset)
            progress.closing()

            print("Neural samples successfully built (the template logic is working correctly)!")
            self._debug_neural_samples(neural_samples)

            # Main training loop
            # TODO: maybe have a validation split to save best model
            # there is a whole range of options for that in the backend, with detailed reporting of the progress across various metrics,
            # but that will only work if the whole training is performed there (i.e. not in python epoch by epoch) and logging turned (to FINE at least)...
            # ...on the other hand it's more flexible to control it from python here, so let's continue with this I guess
            # DZC TODO: try to use the whole training, and logging turned on just for this code segment
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

            print(colored(f"Best model at epoch={best_epoch} with f1_score={best_f1}", "green"))

            # Load the best model state dict
            self.model.load_state_dict(best_state_dict)
        except KeyboardInterrupt:
            print(f"Training stopped early due to keyboard interrupt!")

        # Evaluate again after training
        if self._debug > 1:
            results = self.model(neural_samples, train=False)  # evaluate again due to missing sample outputs
            self._debug_neural_samples(neural_samples, results)
            if self._debug > 2:
                for i in range(len(neural_samples)):
                    result = results[i]
                    sample = neural_samples.samples[i]
                    print(f"target: {sample.target} <- predicted: {result} for sample: {sample.java_sample.query.ID}")

        print("-" * 80)

    def _debug_neural_samples(self, neural_samples, results=None):
        """Check that there are no problems and show some statistics"""
        if not results:
            outputs = [0] * len(neural_samples)
        else:
            outputs = results

        state2actions = {}
        for output, sample in zip(outputs, neural_samples):
            state_net = sample.java_sample.query.evidence.getId()  # a shared neural net for the given State
            actions = state2actions.get(state_net, [])
            actions.append((sample, sample.target, sample.java_sample.query.neuron, output))
            state2actions[state_net] = actions
        num_reachable_negative = 0
        num_multiple = 0
        num_reachable_positive = 0
        correctly_ordered = 0
        for state, actions in state2actions.items():
            reachable_actions = []
            reachable_positive = []
            reachable_negative = []
            for action in actions:
                if action[2]:
                    reachable_actions.append(action)
                else:
                    continue

                if action[1]:
                    reachable_positive.append(action)
                else:
                    reachable_negative.append(action)

            # some debugging to check whether optimal actions are preserved
            debug_colour = "magenta"
            if not reachable_actions and self._debug > 1:
                print(colored(f"State {state} has no reachable actions at all!", debug_colour))
            if not reachable_positive and self._debug > 1:
                actions = [(action[0].java_sample.query.ID.split(":")[1], int(action[1]), action[0].java_sample.query.ID.split(":")[0]) for action in actions]
                sid = actions[0][2]
                print(colored(f"State {state} {sid} has no reachable positive (optimal) actions at all!", debug_colour))
                print(colored("Applicable actions:", debug_colour))
                for action_name, is_optimal, _ in sorted(actions, key=lambda item: repr(item[1]) + repr(item[0]), reverse=True):
                    print(colored(f"\t{is_optimal} : {action_name}", debug_colour))
                print(colored("Reachable actions:", debug_colour))
                for action in reachable_actions:
                    action_name = action[0].java_sample.query.ID.split(":")[1]
                    print(colored(f"\t{action_name}", debug_colour))

            if len(reachable_negative) > 0:
                num_reachable_negative += 1
            if len(reachable_actions) > 1:
                num_multiple += 1
            if len(reachable_positive) > 0:
                num_reachable_positive += 1

            if results and len(reachable_actions) > 0:
                ordered = sorted(reachable_actions, key=lambda item: item[3], reverse=True)
                if ordered[0][1] == 1:  # the highest output is for (some) optimal action - good
                    correctly_ordered += 1
                elif ordered[0][1] == 0:  # this is a problematic state (wrongly predicted)...
                    if self._debug > 1:
                        predictions = [(f'{action[1]} : {action[0].java_sample.query.ID} '
                                        f'-> {action[3]}') for action in ordered]
                        print(f'A problematic state to predict an optimal action for: {state} -> {predictions}')

        # fmt: off
        print(f"There are {len(neural_samples)} learning queries across {len(state2actions)} unique states")
        print(f"{float(num_multiple) / len(state2actions) * 100:2f} % of states have more than 1 action derived")
        print(f"{float(num_reachable_positive) / len(state2actions) * 100:2f} % of states have some positive action preserved, ideally this should be 100%")
        print(f"{float(num_reachable_negative) / len(state2actions) * 100:2f} % of states have some negative action derived, and hence can be improved with parameter training")
        if results:
            print(f"{(1 - float(correctly_ordered) / len(state2actions)) * 100:2f} % of states are problematic with wrongly ordered action predictions (suboptimal first before optimal)")
        # fmt: on

    def _grounding_debug(self):
        @jpype.JImplements(jpype.JClass("java.util.function.Consumer"))
        class GroundingCallback:
            def __init__(self):
                self.inference_round = 0
                herbrand_class = jpype.JClass("cz.cvut.fel.ida.logic.subsumption.HerbrandModel")
                herbrand_class.callBack = self

            @jpype.JOverride
            def accept(self, herbrand_model):
                print(f"\n========Herbrand model inference round {self.inference_round}==========\n")
                herbrand_map = sorted(f'{predicate} : {atoms}' for predicate, atoms in herbrand_model.entrySet())
                for line in herbrand_map:
                    print(line)
                self.inference_round += 1

        return GroundingCallback()

    def _grounding_progress(self):
        @jpype.JImplements(jpype.JClass("java.util.function.IntConsumer"))
        class GroundingProgress:
            def __init__(self):
                groundingClass = jpype.JClass("cz.cvut.fel.ida.logic.grounding.GroundTemplate")
                self.samplesClass = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.SamplesBuilder")
                groundingClass.progressBar = self
                self.pbar = None

            @jpype.JOverride
            def accept(self, int):
                if self.pbar is None:
                    self.pbar = tqdm(total=self.samplesClass.counter)
                self.pbar.update(1)

            def closing(self):
                self.pbar.close()

        return GroundingProgress()

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
