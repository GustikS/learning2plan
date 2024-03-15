from typing import List

from code.planning import PlanningState, Action


class Scorer:

    def __init__(self, model):
        self.model = model

    def score_states(self, state: [PlanningState]) -> dict[PlanningState: float]:
        pass

    def score_actions(self, state, actions: [Action]) -> dict[Action: float]:
        pass