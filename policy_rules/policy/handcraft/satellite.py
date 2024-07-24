from neuralogic.core import R, V
from pymimir import Atom
from typing_extensions import override

from ..policy import Policy
from ..policy_learning import FasterLearningPolicy, LearningPolicy

""" 
See satellite_nullary_100.py for version that works on entire state space, 
even if some of these states are not reached by the policy. The version in this
file only derives at least one optimal action for ~79% of states.
"""


class SatellitePolicy(FasterLearningPolicy):
    def print_state(self, state: list[Atom]):
        object_names = sorted([o.name for o in self._problem.objects])
        directions = 0
        instruments = 0
        modes = 0
        satellites = 0
        for o in object_names:
            if o.startswith("d"):
                directions += 1
            elif o.startswith("i"):
                instruments += 1
            elif o.startswith("m"):
                modes += 1
            elif o.startswith("s"):
                satellites += 1

        statics = []
        nonstatics = []
        for atom in state:
            atom = atom.get_name()
            if atom.split("(")[0] in {"calibration_target", "on_board", "supports"}:
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
        print("*" * 80)
        print("Derived predicates:")
        self._debug_inference_helper(R.instrument_config("S", "I", "M"), newline=True)
        self._debug_inference_helper(R.powered_on_focus("S", "I", "M", "D"), newline=True)
        self._debug_inference_helper(R.calibrated_focus("S", "I", "M", "D"), newline=True)
        print("*" * 80)
        print("Actions:")
        self._debug_inference_actions()
        print("=" * 80)

    @override
    def _add_derived_predicates(self):
        """This is a new version that should be easier to debug/maintain"""
        head = R.instrument_config("S", "I", "M")
        body = [
            R.supports("I", "M"),
            R.on_board("I", "S"),
        ]
        self.add_rule(head, body)

        head = R.powered_on_focus("S", "I", "M", "D")
        body = [
            R.ug_have_image("D", "M"),
            R.instrument_config("S", "I", "M"),
            R.power_on("I"),
        ]
        self.add_rule(head, body)

        head = R.calibrated_focus("S", "I", "M", "D")
        body = [
            R.ug_have_image("D", "M"),
            R.instrument_config("S", "I", "M"),
            R.calibrated("I"),
        ]
        self.add_rule(head, body)

        self.add_rule(R.derivable_powered_on_focus, R.powered_on_focus("S", "I", "M", "D"))
        self.add_rule(R.derivable_calibrated_focus, R.calibrated_focus("S", "I", "M", "D"))

        self.add_rule(R.derivable_calibrate, R.calibrate("S", "I", "D"))


    @override
    def _add_policy_rules(self):
        """Facts:
        - calibrated(?i) is always good, it is never a negative precondition
        - power_on(?i) is always good, it is never a negative precondition
        - switch_off(?i, ?s) is needed if a satellite contains several necessary instruments
        - calibrate only has to be done once for each necessary instrument
        """

        # 1. switch on instrument in a satellite that supports the goal mode
        body = [
            R.ug_have_image("D_goal", "M"),
            R.instrument_config("S", "I", "M"),
            ~R.derivable_powered_on_focus,
        ]
        self.add_output_action("switch_on", body)

        # 2a. turn the satellite to calibration target (if necessary)
        body = [
            R.powered_on_focus("S", "I", "M", "D_goal"),
            ~R.calibrated("I"),
            ~R.pointing("S", "D_new"),  # D_new = calibration direction to turn_to
            ~R.derivable_calibrate,
        ]
        self.add_output_action("turn_to", body)

        # 2b. calibrate instrument
        body = [
            R.powered_on_focus("S", "I", "M", "D_goal"),
            ~R.calibrated("I"),
        ]
        self.add_output_action("calibrate", body)

        # 3a. turn to goal direction (if necessary)
        body = [
            R.calibrated_focus("S", "I", "M", "D_new"),
        ]
        self.add_output_action("turn_to", body)

        # 3b. take image
        body = [
            R.calibrated_focus("S", "I", "M", "D"),
        ]
        self.add_output_action("take_image", body)

        # 4. deal with pointing goals
        body = [
            R.ug_pointing("S", "D_new"),
            # (LP)
            ~R.derivable_calibrated_focus,
            ~R.derivable_powered_on_focus,
        ]
        self.add_output_action("turn_to", body)
