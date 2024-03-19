from abc import ABC
from typing import List

from learning2plan.learning.modelsTorch import get_compatible_model, get_tensor_dataset, get_predictions_torch
from learning2plan.planning import PlanningState, Action


class Scorer(ABC):

    def __init__(self, model_type, encoding_type, backend, instance):
        self.model_type = model_type
        self.encoding_type = encoding_type
        self.backend = backend
        self.instance = instance

    def score_states(self, states: [PlanningState]) -> {PlanningState: float}:
        pass

    def score_actions(self, backend_state, frontend_state, backend_ground_actions,
                      frontend_ground_actions: [Action]) -> {Action: float}:
        pass


class TorchScorer(Scorer):

    def __init__(self, model_type, encoding_type, backend, instance):
        super().__init__(model_type, encoding_type, backend, instance)
        self.model = None

    def score_states(self, states: [PlanningState]) -> {PlanningState: float}:
        samples = []
        for state in states:
            samples.append(state.get_sample(self.encoding_type))

        if not self.model:
            self.model = get_compatible_model(samples, model_class=self.model_type, num_layers=2, hidden_channels=8)

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

    def __init__(self, model_type, encoding_type, backend, instance):
        super().__init__(model_type, encoding_type, backend, instance)
        self.model = None

    def score_states(self, states: [PlanningState]) -> {PlanningState: float}:
        samples = []
        for state in states:
            samples.append(state.get_sample(self.encoding_type))

    def score_actions(self, backend_state, frontend_state, backend_ground_actions,
                      frontend_ground_actions: [Action]) -> {Action: float}:
        # next_states = []
        # for action in actions:
        #     next_state = self.backend.planner.nextState(state, action)
        #     next_sample = next_state.get_sample(self.encoding)

        # todo call the model here
        return {action: 1.0 for action in frontend_ground_actions}
