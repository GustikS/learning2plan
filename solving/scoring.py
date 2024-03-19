from abc import ABC
from typing import List

from learning2plan.learning.modelsTorch import get_compatible_model
from learning2plan.planning import PlanningState, Action


class Scorer(ABC):

    def __init__(self, model_type, encoding_type, backend):
        self.model_type = model_type
        self.encoding_type = encoding_type
        self.backend = backend

    def score_states(self, states: [PlanningState]) -> {PlanningState: float}:
        pass

    def score_actions(self, state, actions: [Action]) -> {Action: float}:
        pass


class TorchScorer(Scorer):

    def __init__(self, model_type, encoding_type, backend):
        super().__init__(model_type, encoding_type, backend)
        self.model = None

    def score_states(self, states: [PlanningState]) -> {PlanningState: float}:
        samples = []
        for state in states:
            samples.append(state.get_sample(self.encoding_type))

        if not self.model:
            self.model = get_compatible_model(samples, model_class=self.model_type, num_layers=2, hidden_channels=8)

    def score_actions(self, state, actions: [Action]) -> {Action: float}:
        # next_states = []
        # for action in actions:
        #     next_state = self.backend.planner.nextState(state, action)
        #     next_sample = next_state.get_sample(self.encoding)

        # todo call the model here
        return {action: 1.0 for action in actions}


class LRNNScorer(Scorer):

    def __init__(self, model_type, encoding_type, backend):
        super().__init__(model_type, encoding_type, backend)
        self.model = None

    def score_states(self, states: [PlanningState]) -> {PlanningState: float}:
        samples = []
        for state in states:
            samples.append(state.get_sample(self.encoding_type))


    def score_actions(self, state, actions: [Action]) -> {Action: float}:
        # next_states = []
        # for action in actions:
        #     next_state = self.backend.planner.nextState(state, action)
        #     next_sample = next_state.get_sample(self.encoding)

        # todo call the model here
        return {action: 1.0 for action in actions}
