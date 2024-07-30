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
        # self._debug_inference_helper(R.phase_one("S", "I", "M", "D"), newline=True)
        # self._debug_inference_helper(R.phase_two("S", "I", "M", "D"), newline=True)
        print("*" * 80)
        print("Actions:")
        self._debug_inference_actions()
        print("=" * 80)

    @override
    def _add_derived_predicates(self):
        """This is a new version that should be easier to debug/maintain"""
        head = R.instrument_config("S", "I", "M", "D_goal")
        body = [
            R.ug_have_image("D_goal", "M"),
            R.supports("I", "M"),
            R.on_board("I", "S"),
        ]
        self.add_rule(head, body)

        self.add_rule(R.derivable_calibrate, R.calibrate("S", "I", "D"))
        self.add_rule(R.derivable_take_image, R.take_image("S", "D", "I", "M"))
        self.add_rule(R.derivable_ug_have_image, R.ug_have_image("D", "M"))

        # head = R.phase_one("S", "I", "M", "D")
        # body = [
        #     R.ug_have_image("D", "M"),
        #     R.instrument_config("S", "I", "M"),
        #     R.power_on("I"),
        # ]
        # self.add_rule(head, body)

        # head = R.phase_two("S", "I", "M", "D")
        # body = [
        #     R.ug_have_image("D", "M"),
        #     R.instrument_config("S", "I", "M"),
        #     R.calibrated("I"),
        # ]
        # self.add_rule(head, body)

        # self.add_rule(R.derivable_phase_one, R.phase_one("S", "I", "M", "D"))
        # self.add_rule(R.derivable_phase_two, R.phase_two("S", "I", "M", "D"))


    @override
    def _add_policy_rules(self):
        """ turn_to(?s - satellite ?d_new - direction ?d_prev - direction) """
        """ switch_on(?i - instrument ?s - satellite) """
        """ switch_off(?i - instrument ?s - satellite) """
        """ calibrate(?s - satellite ?i - instrument ?d - direction) """
        """ take_image(?s - satellite ?d - direction ?i - instrument ?m - mode) """

        # # 1a. switch off instrument if necessary
        # body = []
        # self.add_output_action("switch_off", body)

        # 1b. switch on instrument in a satellite that supports the goal mode
        body = [
            R.instrument_config("S", "I", "M", "D_goal"),
        ]
        self.add_output_action("switch_on", body)

        # 2a. turn the satellite to calibration target (if necessary)
        body = [
            R.instrument_config("S", "I", "M", "D_goal"),
            R.calibration_target("I", "D_new"),  # D_new = calibration direction to turn_to
            ~R.calibrated("I"),
            ~R.pointing("S", "D_new"),
            # ~R.derivable_calibrate,
            ~R.derivable_take_image,
        ]
        self.add_output_action("turn_to", body)

        # 2b. calibrate instrument
        body = [
            R.instrument_config("S", "I", "M", "D_goal"),
            ~R.calibrated("I"),
        ]
        self.add_output_action("calibrate", body)

        # 3a. turn to goal direction (if necessary)
        body = [
            R.instrument_config("S", "I", "M", "D_new"),
            R.calibrated("I"),
            # ~R.derivable_calibrate,
            ~R.derivable_take_image,
        ]
        self.add_output_action("turn_to", body)

        # 3b. take image
        body = [
            R.instrument_config("S", "I", "M", "D"),
        ]
        self.add_output_action("take_image", body)

        # 4. deal with pointing goals
        body = [
            R.ug_pointing("S", "D_new"),
            # (LP)
            ~R.derivable_ug_have_image,
        ]
        self.add_output_action("turn_to", body)
