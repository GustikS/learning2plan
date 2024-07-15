from neuralogic.core import Aggregation, C, Combination, Metadata, R, Template, Transformation, V
from pymimir import Atom
from typing_extensions import override

from ..policy import Policy
from ..policy_learning import FasterLearningPolicy, LearningPolicy


class TransportPolicy(FasterLearningPolicy):
    def print_state(self, state: list[Atom]):
        object_names = sorted([o.name for o in self._problem.objects])
        statics = []
        nonstatics = []
        for atom in state:
            atom = atom.get_name()
            if atom.split("(")[0] in {"road"}:
                statics.append(atom)
            else:
                nonstatics.append(atom)

        nonstaticsset = set(nonstatics)
        print()
        print("Statics:")
        for f in sorted(statics):
            print(f)

        print()
        print("Goal:")
        goals = []
        for g in self._goal:
            assert not g.negated
            goals.append(g.atom.get_name())
        for g in sorted(goals):
            if g in nonstaticsset:
                g += " *"
            else:
                g += " x"
            print(g)

        print()
        print("Current state:")
        for f in sorted(nonstatics):
            print(f)

        self._prev_state = state

    def _debug_inference(self):
        print("Inference for current state:")
        self._debug_inference_actions()
        self._debug_inference_helper(R.shortest("A", "B"), newline=True)
        self._debug_inference_helper(R.derivable_pickup, newline=True)
        self._debug_inference_helper(R.derivable_drop, newline=True)
        print("=" * 80)

    @override
    def _add_derived_predicates(self):
        self.add_rule(R.derivable_pickup, R.pickup("V", "L", "P", "S1", "S2"))
        self.add_rule(R.derivable_drop, R.drop("V", "L", "P", "S1", "S2"))

        # DZC 15/07/2024 TODOs: 
        # - code to initialise weights to start from 1 and have shortest path based on unit costs so we are not just aggregating randomly initialised weights
        # - understand Metadata(combination) to compute closest location from a given location
        metadata = Metadata(aggregation=Aggregation.MIN, transformation=Transformation.IDENTITY)
        self._template += (R.shortest("L1", "L2") <= R.road("L1", "L2")) | metadata
        self._template += (R.shortest("L1", "L2") <= (R.road("L1", "L3"), R.shortest("L3", "L2"))) | metadata

    @override
    def _add_policy_rules(self):
        """ pickup(?v - vehicle ?l - location ?p - package ?s1 ?s2 - size) """
        # pick up any package not at goal location
        body = [
            R.ug_at("P", "Goal_location"),
        ]
        self.add_output_action("pickup", body)

        """ drop(?v - vehicle ?l - location ?p - package ?s1 ?s2 - size) """
        # drop any package at goal location
        body = [
            R.ug_at("P", "L"),
        ]
        self.add_output_action("drop", body)

        """ drive(?v - vehicle ?l1 ?l2 - location) """
        # last priority for drive (check if can pick up and drop anything first)

        # drive vehicle to nearest goal package to pick up
        body = [
            R.ap_at("P", "Loc_of_package"),
            R.ug_at("P", "Goal_location"),
            R.shortest("L2", "Loc_of_package"),
            ~R.derivable_pickup,
            ~R.derivable_drop,
        ]
        self.add_output_action("drive", body)

        # drive vehicle to nearest goal location to drop off package
        body = [
            R.ap_in("P", "V"),  # package in truck
            R.ug_at("P", "Goal_location"),
            R.shortest("L2", "Goal_location"),
            ~R.derivable_pickup,
            ~R.derivable_drop,
        ]
        self.add_output_action("drive", body)
