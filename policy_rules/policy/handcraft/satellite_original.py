from neuralogic.core import R
from pymimir import Atom
from typing_extensions import override

from ..policy import Policy
from ..policy_learning import \
    LearningPolicy  # better this supervised (slow) version due to the unstable negation


class SatellitePolicy(LearningPolicy):
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
        self._debug_inference_helper(R.instrument_config("S", "I", "M"), newline=True)
        self._debug_inference_helper(R.exists_ug_have_image("S"), newline=True)
        self._debug_inference_helper(R.exists_calibrate("S"), newline=True)
        self._debug_inference_helper(R.exists_take_image("S"), newline=True)
        self._debug_inference_helper(R.exists_switch_on("S"), newline=True)
        print("-" * 80)
        self._debug_inference_actions()
        print("=" * 80)

    @override
    def _add_derived_predicates(self):
        head = R.instrument_config("S", "I", "M")
        body = [
            R.supports("I", "M"),
            R.on_board("I", "S"),
        ]
        self.add_rule(head, body)

        # "S" is dummy in the following since neuralogic does not have nullary predicates
        # It does have nullary predicates - please see the nullary version also for tricks with negation

        head = R.exists_ug_have_image("S")
        body = [
            R.ug_have_image("D", "M"),
            R.satellite("S"),
        ]
        self.add_rule(head, body)

        head = R.exists_calibrate("S")
        body = [R.calibrate("S_other", "I", "D"), R.satellite("S")]
        self.add_rule(head, body)

        head = R.exists_take_image("S")
        body = [R.take_image("S_other", "D", "I", "M"), R.satellite("S")]
        self.add_rule(head, body)

        head = R.exists_switch_on("S")
        body = [R.switch_on("I", "S_other"), R.satellite("S")]
        self.add_rule(head, body)

        head = R.exists_turn_to_ug_have_image("S")
        body = [R.turn_to_ug_have_image("S_other", "D_new", "D_prev"), R.satellite("S")]
        self.add_rule(head, body)

    @override
    def _add_policy_rules(self):
        """Facts:
        - calibrated(?i) is always good, it is never a negative precondition
        - power_on(?i) is always good, it is never a negative precondition
        - switch_off(?i, ?s) is needed if a satellite contains several necessary instruments
        - calibrate only has to be done once for each necessary instrument
        """

        """ turn_to(?s - satellite ?d_new - direction ?d_prev - direction) """
        # Ensure turn_to is always last priority (LP)

        # turn towards unachieved have_image goals
        body = [
            R.ug_have_image("D_new", "M"),
            R.instrument_config("S", "I", "M"),
            R.calibrated("I"),
            # (LP)
            ~R.exists_calibrate("S"),
            ~R.exists_take_image("S"),
            ~R.exists_switch_on("S"),
        ]
        head = R.turn_to_ug_have_image("S", "D_new", "D_prev")
        self.add_rule(head, body)
        self.add_output_action("turn_to", [head])

        # turn towards calibration direction if instrument is not turned on
        body = [
            R.ug_have_image("D_other", "M"),
            R.instrument_config("S", "I", "M"),
            ~R.calibrated("I"),
            R.calibration_target("I", "D_new"),
            ~R.exists_turn_to_ug_have_image("S"),
            # (LP)
            ~R.exists_calibrate("S"),
            ~R.exists_take_image("S"),
            ~R.exists_switch_on("S"),
        ]
        self.add_output_action("turn_to", body)

        # for pointing goals
        body = [
            R.ug_pointing("S", "D_new"),
            ~R.exists_ug_have_image("S"),
            ~R.exists_turn_to_ug_have_image("S"),
            # (LP)
            ~R.exists_calibrate("S"),
            ~R.exists_take_image("S"),
            ~R.exists_switch_on("S"),
        ]
        self.add_output_action("turn_to", body)

        """ switch_on(?i - instrument ?s - satellite) """
        # switch on any instrument that may contribute towards the goal
        body = [
            R.ug_have_image("D", "M"),
            R.instrument_config("S", "I", "M"),
            ~R.power_on("I"),
            ~R.calibrated("I"),
        ]
        self.add_output_action("switch_on", body)

        """ switch_off(?i - instrument ?s - satellite) """
        # switch off any instrument that is not needed, and another instrument is needed
        body = [
            ~R.ug_have_image("I_other", "M"),
            R.instrument_config("S", "I_other", "M"),
            ~R.calibrated("I_other"),
        ]

        """ calibrate(?s - satellite ?i - instrument ?d - direction) """
        body = [
            R.ug_have_image("D", "M"),
            R.pointing("S", "D"),
            R.instrument_config("S", "I", "M"),
            ~R.calibrated("I"),
        ]
        self.add_output_action("calibrate", body)

        """ take_image(?s - satellite ?d - direction ?i - instrument ?m - mode) """
        body = [
            R.ug_have_image("D", "M"),
            R.instrument_config("S", "I", "M"),
        ]
        self.add_output_action("take_image", body)