from neuralogic.core import R
from pymimir import Atom
from typing_extensions import override
from util.printing import print_mat

from ..policy import Policy
from ..policy_learning import LearningPolicy, FasterLearningPolicy


class MiconicPolicy(FasterLearningPolicy):
    def print_state(self, state: list[Atom]):
        object_names = sorted([o.name for o in self._problem.objects])
        floors = 0
        passengers = 0
        for o in object_names:
            if o.startswith("f"):
                floors += 1
            elif o.startswith("p"):
                passengers += 1
        
        items_at_floor_goals = {f"f{f}": [] for f in list(range(1, floors + 1))}
        items_at_floor = {f"f{f}": [] for f in list(range(1, floors + 1))}
        items_at_floor["L"] = []
        
        for atom in state:
            pred_name = atom.predicate.name
            if pred_name == "lift-at":
                items_at_floor[atom.terms[0].name].append("L")
            elif pred_name == "destin":
                items_at_floor_goals[atom.terms[1].name].append(atom.terms[0].name)
            elif pred_name == "origin":
                items_at_floor[atom.terms[1].name].append(atom.terms[0].name)
            elif pred_name == "boarded":
                items_at_floor["L"].append(atom.terms[0].name)
        
        print()
        print("Goal state:")
        for i in range(floors, 0, -1):
            f = f"f{i}"
            print(f"{i} | {' '.join(sorted(items_at_floor_goals[f]))}")
        
        print()
        print("Current state:")
        print(f"  L --[ {' '.join(sorted(items_at_floor['L']))}")
        for i in range(floors, 0, -1):
            f = f"f{i}"
            print(f"{i} | {' '.join(sorted(items_at_floor[f]))}")

        self._prev_state = state

    def _debug_inference(self):
        print("Inference for current state:")
        self._debug_inference_actions()
        print("=" * 80)

    @override
    def _add_derived_predicates(self):
        pass

    @override
    def _add_policy_rules(self):
        """ board(?f - floor ?p - passenger) """
        body = []
        self.add_output_action("board", body)


        """ depart(?f - floor ?p - passenger) """
        body = [
            R.get("ug_served")("P"),
        ]
        self.add_output_action("depart", body)


        """ up(?f1 - floor ?f2 - floor) """
        # [cannot board anymore and go to a passenger not at goal]
        body = [
            ~R.get("board")("F1", "P1"),
            R.get("ug_served")("P2"),
            R.get("origin")("P2", "F2"),
            R.get("destin")("P2", "F3"),
        ]
        self.add_output_action("up", body)

        # [cannot board anymore and go to a boarded passenger's goal location]
        body = [
            ~R.get("board")("F1", "P1"),
            R.get("ug_served")("P2"),
            R.get("boarded")("P2"),
            R.get("destin")("P2", "F2"),
        ]
        self.add_output_action("up", body)


        """ down(?f1 - floor ?f2 - floor) """  # exact same rules as up
        body = [
            ~R.get("board")("F1", "P"),
            R.get("ug_served")("P2"),
            R.get("origin")("P2", "F2"),
            R.get("destin")("P2", "F3"),
        ]
        self.add_output_action("down", body)

        body = [
            ~R.get("board")("F1", "P1"),
            R.get("ug_served")("P2"),
            R.get("boarded")("P2"),
            R.get("destin")("P2", "F2"),
        ]
        self.add_output_action("down", body)
