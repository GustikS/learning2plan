from neuralogic.core import R
from typing_extensions import override

from ..policy import Policy


class SatellitePolicy(Policy):
    @override
    def _add_derived_predicates(self):
        head = R.instrument_config("I", "M", "S", "D")
        body = [
            R.supports("I", "M"),
            R.on_board("I", "S"),
            R.calibration_target("I", "D"),
        ]
        self._template += head <= body

    @override
    def _add_policy_rules(self):
        """ turn_to(?s - satellite ?d_new - direction ?d_prev - direction) """
        body = [
            R.ug_have_image("D_new", "M"),
            R.instrument_config("I", "M", "S", "D_new"),
        ]
        self.add_rule("turn_to", body)

        # for `pointing` goals
        body = [
            R.ug_pointing("S", "D_new"),
            ~R.ug_have_image("D_other", "M_other"),
            R.instrument_config("I", "M", "S", "D_other"),
        ]
        self.add_rule("turn_to", body)

        """ switch_on(?i - instrument ?s - satellite) """
        body = [
            R.ug_have_image("D", "M"),
            R.pointing("S", "D"),
            R.instrument_config("I", "M", "S", "D"),
            ~R.power_on("I"),
        ]
        self.add_rule("switch_on", body)

        """ switch_off(?i - instrument ?s - satellite) """
        # TODO

        """ calibrate(?s - satellite ?i - instrument ?d - direction) """
        body = [
            R.ug_have_image("D", "M"),
            R.pointing("S", "D"),
            R.instrument_config("I", "M", "S", "D"),
            ~R.calibrated("I"),
        ]
        self.add_rule("calibrate", body)

        """ take_image(?s - satellite ?d - direction ?i - instrument ?m - mode) """
        body = [
            R.ug_have_image("D", "M"),
            R.instrument_config("I", "M", "S", "D"),
        ]
        self.add_rule("take_image", body)
