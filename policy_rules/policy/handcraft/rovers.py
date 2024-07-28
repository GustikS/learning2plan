from itertools import combinations, product

import networkx as nx
from neuralogic.core import Aggregation, C, Combination, Metadata, R, Template, Transformation, V
from pymimir import Atom, Domain, Problem
from typing_extensions import override

from ..policy import Policy
from ..policy_learning import FasterLearningPolicy, LearningPolicy


class RoversPolicy(FasterLearningPolicy):
    def __init__(self, domain: Domain, debug=0):
        super().__init__(domain, debug)
        self._statics = {
            "calibration_target",
            "on_board",
            "supports",
            "visible",
            "visible_from",
            "store_of",
            "equipped_for_soil_analysis",
            "equipped_for_rock_analysis",
            "equipped_for_imaging",
        }

    def _debug_inference(self):
        print("Inference for current state:")
        # super()._debug_inference()
        # self._grounding_debug()
        # self._debug_inference_actions()
        self._debug_inference_helper(R.uncollected_goal_soil("P"), newline=True)
        self._debug_inference_helper(R.uncollected_goal_rock("P"), newline=True)
        self._debug_inference_helper(R.navigate("X", "Y", "Z"), newline=True)

        print("=" * 80)
        # breakpoint()

    @override
    def _add_derived_predicates(self):
        head = R.collected_rock("P")
        body = [
            R.have_rock_analysis("R", "P"),
        ]
        self.add_rule(head, body)

        head = R.uncollected_goal_rock("P")
        body = [
            R.ug_communicated_rock_data("P"),
            # ~R.collected_rock("P"),
        ]
        self.add_rule(head, body)

        head = R.collected_soil("P")
        body = [
            R.have_soil_analysis("R", "P"),
        ]
        self.add_rule(head, body)

        head = R.uncollected_goal_soil("P")
        body = [
            R.ug_communicated_soil_data("P"),
            # ~R.collected_soil("P"),
        ]
        self.add_rule(head, body)

        head = R.uncollected_image("O", "M")
        body = [
            R.have_image("R", "O", "M"),
        ]
        self.add_rule(head, body)

        head = R.uncollected_goal_image("O", "M")
        body = [
            R.ug_communicated_image_data("O", "M"),
            # ~R.uncollected_image("O", "M"),
        ]
        self.add_rule(head, body)

        # self.add_rule(R.derivable_sample_soil, R.sample_soil("X", "S", "P"))
        # self.add_rule(R.derivable_sample_rock, R.sample_rock("X", "S", "P"))
        # self.add_rule(R.derivable_drop, R.drop("X", "Y"))
        # self.add_rule(R.derivable_calibrate, R.calibrate("R", "I", "T", "W"))
        # self.add_rule(R.derivable_take_image, R.take_image("R", "P", "O", "I", "M"))
        # self.add_rule(R.derivable_communicate_soil_data, R.communicate_soil_data("R", "L", "P", "X", "Y"))
        # self.add_rule(R.derivable_communicate_rock_data, R.communicate_rock_data("R", "L", "P", "X", "Y"))
        # self.add_rule(R.derivable_communicate_image_data, R.communicate_image_data("R", "L", "O", "M", "X", "Y"))

        head = R.supports_mode("R", "I", "M")
        body = [
            R.on_board("I", "R"),
            R.equipped_for_imaging("R"),
            R.supports("I", "M"),
        ]
        self.add_rule(head, body)

        # head = R.derivable_calibrated_and_supports_mode("M")
        # body = [
        #     R.calibrated("I", "R"),
        #     R.supports_mode("R", "I", "M"),
        # ]
        # self.add_rule(head, body)

    @override
    def _add_policy_rules(self):
        # https://ipc2023-learning.github.io/talk.pdf
        # 1. for each rock/soil data in the goal, get a rover equipped for
        # rock/soil analysis and can move to that waypoint, sample and drop it

        """navigate(?x - rover ?y - waypoint ?z - waypoint)"""

        navigate_priority = [
            # ~R.at("X", "Z"),  # domain misses precondition that we don't go to the same place
            # ~R.derivable_sample_soil,
            # ~R.derivable_sample_rock,
            # ~R.derivable_calibrate,
            # ~R.derivable_take_image,
        ]

        # move to uncollected goal soil
        body = navigate_priority
        body += [
            R.uncollected_goal_soil("Z"),
            R.at_soil_sample("Z"),
            R.equipped_for_soil_analysis("X"),
        ]
        head = R.navigate_to_soil("X", "Z")
        self.add_rule(head, body)
        self.add_output_action("navigate", [head])
        # self.add_output_action("navigate", body)

        # move to uncollected goal rock
        body = navigate_priority
        body += [
            R.uncollected_goal_rock("Z"),
            R.at_rock_sample("Z"),
            R.equipped_for_rock_analysis("X"),
        ]
        head = R.navigate_to_rock("X", "Z")
        self.add_rule(head, body)
        self.add_output_action("navigate", [head])
        # self.add_output_action("navigate", body)

        # # move to uncalibrated calibration subgoal
        # body = navigate_priority
        # body += [
        #     R.uncollected_goal_image("O", "M"),
        #     R.supports_mode("X", "I", "M"),
        #     R.calibration_target("I", "T"),
        #     R.visible_from("T", "Z")
        # ]
        # self.add_output_action("navigate", body)

        # # move to uncollected goal image
        # body = navigate_priority
        # body += [
        #     R.uncollected_goal_image("O", "M"),
        #     R.supports_mode("X", "I", "M"),
        #     R.calibrated("I", "X"),
        #     R.visible_from("O", "Z"),
        # ]
        # self.add_output_action("navigate", body)

        # # move to communicate soil data
        # body = navigate_priority
        # body += [
        #     R.ug_communicated_soil_data("P"),
        #     R.have_soil_analysis("X", "P"),
        #     R.at_lander("L", "Z1"),
        #     R.visible("Z", "Z1"),
        # ]
        # self.add_output_action("navigate", body)

        # # move to communicate rock data
        # body = navigate_priority
        # body += [
        #     R.ug_communicated_rock_data("P"),
        #     R.have_rock_analysis("X", "P"),
        #     R.at_lander("L", "Z1"),
        #     R.visible("Z", "Z1"),
        # ]
        # self.add_output_action("navigate", body)

        # # move to communicate image data
        # body = navigate_priority
        # body += [
        #     R.ug_communicated_image_data("P"),
        #     R.have_image("X", "P"),
        #     R.at_lander("L", "Z1"),
        #     R.visible("Z", "Z1"),
        # ]
        # self.add_output_action("navigate", body)

        """ sample_soil(?x - rover ?s - store ?p - waypoint) """
        body = [
            R.uncollected_goal_soil("P"),
        ]
        self.add_output_action("sample_soil", body)

        """ sample_rock(?x - rover ?s - store ?p - waypoint) """
        body = [
            R.uncollected_goal_rock("P"),
        ]
        self.add_output_action("sample_rock", body)

        """ drop(?x - rover ?y - store) """
        body = []
        self.add_output_action("drop", body)

        """ calibrate(?r - rover ?i - camera ?t - objective ?w - waypoint) """
        # body = [
        #     # R.supports_mode("R", "I", "M"),
        #     # R.usg_image("O", "M"),
        #     # ~R.derivable_calibrated_and_supports_mode("M"),
        # ]
        # self.add_output_action("calibrate", body)

        """ take_image(?r - rover ?p - waypoint ?o - objective ?i - camera ?m - mode) """
        body = [
            R.uncollected_goal_image("O", "M"),
        ]
        self.add_output_action("take_image", body)

        body = [
            R.ug_communicated_soil_data("P"),
        ]
        self.add_output_action("communicate_soil_data", body)

        body = [
            R.ug_communicated_rock_data("P"),
        ]
        self.add_output_action("communicate_rock_data", body)

        body = [
            R.ug_communicated_image_data("O", "M"),
        ]
        self.add_output_action("communicate_image_data", body)
