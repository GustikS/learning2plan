from abc import ABC
from typing import List

from learning2plan.learning.modelsLRNN import LRNN, get_relational_dataset, get_predictions_LRNN
from learning2plan.learning.modelsTorch import get_compatible_model, get_tensor_dataset, get_predictions_torch
from learning2plan.planning import PlanningState, Action


class Scorer(ABC):

    def __init__(self, model, encoding_type, backend, instance):
        self.model = model
        self.encoding_type = encoding_type
        self.backend = backend
        self.instance = instance

    def score_states(self, states: [PlanningState]) -> {PlanningState: float}:
        pass

    def score_actions(self, backend_state, frontend_state, backend_ground_actions,
                      frontend_ground_actions: [Action]) -> {Action: float}:
        pass


class TorchScorer(Scorer):

    def __init__(self, model, encoding_type, backend, instance):
        super().__init__(model, encoding_type, backend, instance)

    def score_states(self, states: [PlanningState]) -> {PlanningState: float}:
        samples = []
        for state in states:
            samples.append(state.get_sample(self.encoding_type))

        tensor_dataset = get_tensor_dataset(samples)
        return get_predictions_torch(self.model, tensor_dataset, reset_weights=False)

    def score_actions(self, backend_state, frontend_state, backend_ground_actions,
                      frontend_ground_actions: [Action]) -> {Action: float}:
        successor_states = []
        for ground_action in backend_ground_actions:
            next_state = self.backend.planner.nextState(backend_state, ground_action)
            successor_states.append(PlanningState.from_backend(next_state, self.instance.domain))
        scores = self.score_states(successor_states)
        scored_actions = {ground_action: score for (ground_action, score) in zip(backend_ground_actions, scores)}
        sorted_actions = dict(sorted(scored_actions.items(), key=lambda item: item[1]))
        return sorted_actions


class LRNNScorer(Scorer):

    def __init__(self, model, encoding_type, backend, instance):
        super().__init__(model, encoding_type, backend, instance)

    def score_states(self, states: [PlanningState]) -> {PlanningState: float}:
        samples = []
        for state in states:
            samples.append(state.get_sample(self.encoding_type))

        relational_dataset = get_relational_dataset(samples)
        built_dataset = self.model.model.build_dataset(relational_dataset)
        return get_predictions_LRNN(self.model, built_dataset, reset_weights=False)

    def score_actions(self, backend_state, frontend_state, backend_ground_actions,
                      frontend_ground_actions: [Action]) -> {Action: float}:
        successor_states = []
        for ground_action in backend_ground_actions:
            next_state = self.backend.planner.nextState(backend_state, ground_action)
            successor_states.append(PlanningState.from_backend(next_state, self.instance.domain))
        scores = self.score_states(successor_states)
        scored_actions = {ground_action: score for (ground_action, score) in zip(backend_ground_actions, scores)}
        sorted_actions = dict(sorted(scored_actions.items(), key=lambda item: item[1]))
        return sorted_actions
