from neuralogic.core import R, V
from pymimir import Atom
from typing_extensions import override

from ..policy import Policy
from ..policy_learning import FasterLearningPolicy, LearningPolicy


class SatellitePolicyNullaryX(FasterLearningPolicy):
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
        self._debug_inference_helper(R.guard_ug_have_image, newline=True)
        self._debug_inference_helper(R.guard_calibrate, newline=True)
        self._debug_inference_helper(R.exists_take_image, newline=True)
        self._debug_inference_helper(R.exists_switch_on, newline=True)
        print("-" * 80)
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

        self.add_rule(R.derivable_ug_have_image, R.ug_have_image("A", "B"))
        self.add_rule(R.guard_ug_have_image, ~R.derivable_ug_have_image, guard_level=3, embedding_layer=-1)

        self.add_rule(R.derivable_calibrate, R.calibrate("A", "B", "C"))
        self.add_rule(R.guard_calibrate, ~R.derivable_calibrate, guard_level=6, embedding_layer=-1)

        self.add_rule(R.derivable_take_image, R.take_image("A", "B", "C", "D"))
        self.add_rule(R.guard_take_image, ~R.derivable_take_image, guard_level=6, embedding_layer=-1)

        self.add_rule(R.derivable_switch_on, R.switch_on("A", "B"))
        self.add_rule(R.guard_switch_on, ~R.derivable_switch_on, guard_level=6, embedding_layer=-1)

        self.add_rule(R.derivable_towards_ug_have_image, R.turn_towards_ug_have_image("A", "B", "C"))
        self.add_rule(R.guard_towards_ug_have_image, ~R.derivable_towards_ug_have_image, guard_level=9, embedding_layer=-1)

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
        # From looking at the entire state space data, there are cases where turn_to is optimal but not calibrate and switch_on/off
        # It may be the case that when actually executing the policy, this problem does not occur...
        # - todo gustav: you can perhaps use some very low weight for that

        # turn towards unachieved have_image goals
        body = [
            R.ug_have_image("D_new", "M"),
            R.instrument_config("S", "I", "M"),
            # R.calibrated("I"),
            # (LP)
            R.guard_take_image,
        ]
        head = R.turn_towards_ug_have_image("S", "D_new", "D_prev")
        self.add_rule(head, body)
        self.add_output_action("turn_to", [head])

        # turn towards calibration direction if instrument is not turned on
        body = [
            R.ug_have_image("D_other", "M"),
            R.instrument_config("S", "I", "M"),
            ~R.calibrated("I"),
            R.calibration_target("I", "D_new"),
            # R.guard_towards_ug_have_image,
            # (LP)
            R.guard_take_image,
        ]
        self.add_output_action("turn_to", body)

        # for pointing goals
        body = [
            R.ug_pointing("S", "D_new"),
            R.guard_ug_have_image,
            R.guard_towards_ug_have_image,
            # (LP)
            R.guard_take_image,
        ]
        self.add_output_action("turn_to", body)

        """ switch_on(?i - instrument ?s - satellite) """
        # switch on any instrument that may contribute towards the goal
        body = [
            R.ug_have_image("D", "M"),
            R.instrument_config("S", "I", "M"),
            ~R.power_on("I"),
            # could be the case intstrument is calibrated but not switched on, so comment this out
            # ~R.calibrated("I"),
        ]
        self.add_output_action("switch_on", body)

        """ switch_off(?i - instrument ?s - satellite) """
        # switch off any instrument that is not needed, and another instrument is needed
        body = [
            ~R.ug_have_image("I_other", "M"),
            R.instrument_config("S", "I_other", "M"),
            ~R.calibrated("I_other"),
        ]
        self.add_output_action("switch_off", body)

        """ calibrate(?s - satellite ?i - instrument ?d - direction) """
        body = [
            R.ug_have_image("D_other", "M"),
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


    def _add_derived_predicates_orig(self):
        head = R.instrument_config("S", "I", "M")
        body = [
            R.supports("I", "M"),
            R.on_board("I", "S"),
        ]
        self.add_rule(head, body)

        # "S" is dummy in the following since neuralogic does not have nullary predicates
        # todo Gustav: it does have nullary predicates - R.nullary
        #  - see modifications/simplifications above, hopefully it is what you meant
        #  - but please be aware that using the "exists..." is a very brittle thing to do
        #       - it will not stop the derivation, if the "exists..." has not been proved YET!
        #           - hence it requires quite some thinking about the derivation order...
        #               - you might use some "guards" for that purpose, e.g. guard_switch <= applicable_switch_on(_,_)
        #                   - and then use turn_on(...) <= guard_switch & !exists_switch_on

        # head = R.exists_ug_have_image("S")
        head = R.exists_ug_have_image
        body = [
            R.ug_have_image("D", "M"),
            # R.satellite("S"),
        ]
        self.add_rule(head, body, is_guard=True)

        # head = R.exists_calibrate("S")
        head = R.exists_calibrate
        body = [
            R.calibrate("S_other", "I", "D"),
            # R.satellite("S")
        ]
        self.add_rule(head, body, is_guard=True)

        # head = R.exists_take_image("S")
        head = R.exists_take_image
        body = [
            R.take_image("S_other", "D", "I", "M"),
            # R.satellite("S")
        ]
        self.add_rule(head, body, is_guard=True)

        # head = R.exists_switch_on("S")
        head = R.exists_switch_on
        body = [
            R.switch_on("I", "S_other"),
            # R.satellite("S")
        ]
        self.add_rule(head, body, is_guard=True)

        # head = R.exists_towards_ug_have_image("S")
        head = R.exists_towards_ug_have_image
        body = [
            R.turn_towards_ug_have_image("S_other", "D_new", "D_prev"),
            # R.satellite("S")
        ]
        self.add_rule(head, body, is_guard=True)
