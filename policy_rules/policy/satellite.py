from neuralogic.core import R
from typing_extensions import override

from .policy import Policy


class SatellitePolicy(Policy):
    @override
    def _add_derived_predicates(self):
        pass

    @override
    def _add_policy_rules(self):
        ## useful macro
        head = R.instrument_config("I", "M", "S", "D")
        body = [
            R.supports("I", "M"),
            R.on_board("I", "S"),
            R.calibration_target("I", "D"),
        ]
        self._template += head <= body

        ## turn_to
        turn_to_head = self.relation_from_schema("turn_to")
        turn_to_rule = self.get_schema_preconditions("turn_to")
        turn_to_rule += [
            R.ug_have_image("D_new", "M"),
            R.instrument_config("I", "M", "S", "D_new"),
        ]
        self._template += turn_to_head <= turn_to_rule

        ## switch_on
        switch_on_head = self.relation_from_schema("switch_on")
        switch_on_rule = self.get_schema_preconditions("switch_on")
        switch_on_rule += [
            R.ug_have_image("D", "M"),
            R.pointing("S", "D"),
            R.instrument_config("I", "M", "S", "D"),
            # ~R.power_on("I"),  ## TODO wait for negative literal fix
            self._get_negative_literal("power_on", ["I"]),
        ]
        self._template += switch_on_head <= switch_on_rule

        ## switch_off
        # TODO

        ## calibrate
        calibrate_head = self.relation_from_schema("calibrate")
        calibrate_rule = self.get_schema_preconditions("calibrate")
        calibrate_rule += [
            R.ug_have_image("D", "M"),
            R.pointing("S", "D"),
            R.instrument_config("I", "M", "S", "D"),
            ~R.calibrated("I"),
        ]
        self._template += calibrate_head <= calibrate_rule

        ## take_image
        take_image_head = self.relation_from_schema("take_image")
        take_image_rule = self.get_schema_preconditions("take_image")
        take_image_rule += [
            R.ug_have_image("D", "M"),
            R.instrument_config("I", "M", "S", "D"),
        ]
        self._template += take_image_head <= take_image_rule

        ## turn_to
        turn_to_head = self.relation_from_schema("turn_to")
        turn_to_rule = self.get_schema_preconditions("turn_to")
        # self._template += R.n_ug_have_image("D_other", "M") <= R.ug_have_image("D_other", "M")
        turn_to_rule += [
            R.ug_pointing("S", "D"),
            R.ag_have_image("D_other", "M"),
            ## we need a not exists feature for this.
            # ~R.ug_have_image("D_other", "M"),  ## TODO wait for negative literal fix
            R.instrument_config("I", "M", "S", "D"),
        ]
        self._template += turn_to_head <= turn_to_rule
