from abc import ABC

import jpype
import neuralogic
from collections import deque

from code.planning import PlanningDataset, PlanningInstance, PlanningState
from code.solving.lrnn import Backend
from code.solving.scoring import Scorer


class Search(ABC):

    def __init__(self, model):
        self.backend = Backend()
        self.matching = self.backend.matching()
        self.get_substitutions = self.backend.planner.getSubstitutions
        self.ground_actions = self.backend.planner.groundActions
        self.next_state = self.backend.planner.nextState

        self.scorer = Scorer(model)

    def run(self, instance: PlanningInstance):
        backend_instance = instance.to_backend(self.backend)
        self.solve(backend_instance)

    def solve(self, backend_instance):
        pass


class Greedy(Search):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def solve(self, instance: PlanningInstance):
        plan = []
        closed = set()

        backend_instance = instance.to_backend(self.backend)
        state = instance.init
        backend_state = state.to_backend(self.backend)

        plan.append((None, state))

        while not backend_instance.isGoal(backend_state, self.matching):
            possible_ground_actions = set()
            for ground_action in backend_instance.actions:
                substitutions = self.get_substitutions(backend_state, ground_action)
                possible_ground_actions.update(self.ground_actions(ground_action, substitutions))

            sorted_action_scores = self.scorer.score_actions(state, possible_ground_actions)   # todo work here with the frontend actions instead?
            for ground_action, action_score in sorted_action_scores:
                next_state = self.next_state(backend_state, ground_action)
                if next_state not in closed:
                    backend_state = next_state
                    state = PlanningState.from_backend(backend_state)
                    closed.add(next_state)  # just check for cycles...
                    plan.append((ground_action, next_state))
                    break
        return plan
