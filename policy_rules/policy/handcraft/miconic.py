from neuralogic.core import R
from pymimir import Atom
from typing_extensions import override

from policy_rules.util.printing import print_mat

from ..policy import Policy
from ..policy_learning import FasterLearningPolicy, LearningPolicy


class MiconicPolicy(FasterLearningPolicy):
    def _print_helper(self, state: list[Atom]):
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

        print(f"  L [ {' '.join(sorted(items_at_floor['L']))} ]")
        for i in range(floors, 0, -1):
            f = f"f{i}"
            print(f"{i} | {' '.join(sorted(items_at_floor[f]))}")

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
        
        if self._prev_state:
            print()
            print("Previous state:")
            self._print_helper(self._prev_state)
        
        print()
        print("Current state:")
        self._print_helper(state)

        self._prev_state = state

    def _debug_inference(self):
        print("Inference for current state:")
        self._debug_inference_actions()
        print("=" * 80)

    @override
    def _add_derived_predicates(self):
        # TODO(DZC): this might make the satisificing policy too strong, look at results to decide whether to keep or not
        self.add_rule(R.derivable_board, R.board("A", "B"))
        self.add_rule(R.derivable_depart, R.depart("A", "B"))

    @override
    def _add_policy_rules(self):
        """ board(?f - floor ?p - passenger) """
        # board whenever possible
        body = []
        self.add_output_action("board", body)


        """ depart(?f - floor ?p - passenger) """
        # depart if passenger at goal location
        body = [
            R.get("ug_served")("P"),
        ]
        self.add_output_action("depart", body)


        """ up(?f1 - floor ?f2 - floor) """
        # [cannot board or depart anymore and go to a passenger not at goal]
        body = [
            ~R.derivable_board,
            ~R.derivable_depart,
            R.get("ug_served")("P2"),
            R.get("origin")("P2", "F2"),
            R.get("destin")("P2", "F3"),
        ]
        self.add_output_action("up", body)

        # [cannot board or depart anymore and go to a boarded passenger's goal location]
        body = [
            ~R.derivable_board,
            ~R.derivable_depart,
            R.get("ug_served")("P2"),
            R.get("boarded")("P2"),
            R.get("destin")("P2", "F2"),
        ]
        self.add_output_action("up", body)


        """ down(?f1 - floor ?f2 - floor) """  # exact same rules as up
        body = [
            ~R.derivable_board,
            ~R.derivable_depart,
            R.get("ug_served")("P2"),
            R.get("origin")("P2", "F2"),
            R.get("destin")("P2", "F3"),
        ]
        self.add_output_action("down", body)

        body = [
            ~R.derivable_board,
            ~R.derivable_depart,
            R.get("ug_served")("P2"),
            R.get("boarded")("P2"),
            R.get("destin")("P2", "F2"),
        ]
        self.add_output_action("down", body)
