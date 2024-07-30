from copy import deepcopy
from itertools import combinations, product

import networkx as nx
from neuralogic.core import Aggregation, C, Combination, Metadata, R, Template, Transformation, V
from pymimir import Atom, Domain, Problem
from typing_extensions import override

from ..policy import Policy
from ..policy_learning import FasterLearningPolicy, LearningPolicy

UNCOLLECTED_GUARD = 3
COLLECT_GUARD = -1
DERIVE_COMM_A_GUARD = 3
DERIVE_COMM_B_GUARD = 6
NAVIGATE_GUARD = 9


class RoversPolicy(FasterLearningPolicy):
    def __init__(self, domain: Domain, debug=0):
        super().__init__(domain, debug)
        self._statics = {
            "at_lander",
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

        # print("=" * 80)
        # self._debug_inference_actions()

        # print("=" * 80)
        # for relation in [
        #     R.collected_rock("P"),
        #     R.uncollected_goal_soil("P"),
        #     R.collected_soil("P"),
        #     R.uncollected_goal_rock("P"),
        #     R.collected_image("O", "M"),
        #     R.uncollected_goal_image("O", "M"),
        # ]:
        #     self._debug_inference_helper(relation, newline=True)

        # print("=" * 80)
        # for relation in [
        #     R.derive_communicate_soil_data,
        #     R.derive_communicate_rock_data,
        #     R.derive_communicate_image_data,
        # ]:
        #     self._debug_inference_helper(relation, newline=True)

        print("=" * 80)
        for relation in [
            R.navigate_to_soil("X", "Z"),
            R.navigate_to_rock("X", "Z"),
            R.navigate_to_calibrate("X", "Z"),
            R.navigate_to_image("X", "Z"),
            R.navigate_to_comm_soil("X", "Z"),
            R.navigate_to_comm_rock("X", "Z"),
            R.navigate_to_comm_image("X", "Z"),
            R.navigate("X", "Y", "Z"),
        ]:
            self._debug_inference_helper(relation, newline=True)
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
            ~R.collected_rock("P"),
        ]
        self.add_rule(head, body, guard_level=UNCOLLECTED_GUARD)

        head = R.collected_soil("P")
        body = [
            R.have_soil_analysis("R", "P"),
        ]
        self.add_rule(head, body)

        head = R.uncollected_goal_soil("P")
        body = [
            R.ug_communicated_soil_data("P"),
            ~R.collected_soil("P"),
        ]
        self.add_rule(head, body, guard_level=UNCOLLECTED_GUARD)

        head = R.collected_image("O", "M")
        body = [
            R.have_image("R", "O", "M"),
        ]
        self.add_rule(head, body)

        head = R.uncollected_goal_image("O", "M")
        body = [
            R.ug_communicated_image_data("O", "M"),
            ~R.collected_image("O", "M"),
        ]
        self.add_rule(head, body, guard_level=UNCOLLECTED_GUARD)

        head = R.supports_mode("R", "I", "M")
        body = [
            R.on_board("I", "R"),
            R.equipped_for_imaging("R"),
            R.supports("I", "M"),
        ]
        self.add_rule(head, body)

        for tup in [
            ("derivable_sample_soil", R.sample_soil("X", "S", "P")),
            ("derivable_sample_rock", R.sample_rock("X", "S", "P")),
            ("derivable_drop", R.drop("X", "Y")),
            ("derivable_calibrate", R.calibrate("R", "I", "T", "W")),
            ("derivable_take_image", R.take_image("R", "P", "O", "I", "M")),
            ("derivable_communicate_soil_data", R.communicate_soil_data("R", "L", "P", "X", "Y"), DERIVE_COMM_A_GUARD),
            ("derivable_communicate_rock_data", R.communicate_rock_data("R", "L", "P", "X", "Y"), DERIVE_COMM_A_GUARD),
            (
                "derivable_communicate_image_data",
                R.communicate_image_data("R", "L", "O", "M", "X", "Y"),
                DERIVE_COMM_A_GUARD,
            ),
            # ("derivable_communicate", R.derivable_communicate_soil_data, DERIVE_COMM_B_GUARD),
            # ("derivable_communicate", R.derivable_communicate_rock_data, DERIVE_COMM_B_GUARD),
            # ("derivable_communicate", R.derivable_communicate_image_data, DERIVE_COMM_B_GUARD),
        ]:
            if len(tup) == 2:
                head, body = tup
                guard_level = -1
            else:
                assert len(tup) == 3
                head, body, guard_level = tup
            self.add_rule(R.get(head), [body], guard_level=guard_level)

        head = R.derivable_calibrated_and_supports_mode("M")
        body = [
            R.calibrated("I", "R"),
            R.supports_mode("R", "I", "M"),
        ]
        self.add_rule(head, body)

    @override
    def _add_policy_rules(self):
        # https://ipc2023-learning.github.io/talk.pdf
        # 1. for each rock/soil data in the goal, get a rover equipped for
        # rock/soil analysis and can move to that waypoint, sample and drop it

        """navigate(?x - rover ?y - waypoint ?z - waypoint)"""

        navigate_priority = [
            # Tne domain misses precondition that we don't go to the same place.
            ~R.at("X", "Z"),
            # The following prioritises other actions over navigating
            ~R.derivable_sample_soil,
            ~R.derivable_sample_rock,
            # ~R.derivable_drop,
            ~R.derivable_calibrate,
            ~R.derivable_take_image,
            ~R.derivable_communicate_soil_data,
            ~R.derivable_communicate_rock_data,
            ~R.derivable_communicate_image_data,
        ]

        # move to uncollected goal soil
        body = deepcopy(navigate_priority)  # Python being annoying with copying
        body += [
            R.uncollected_goal_soil("Z"),
            R.at_soil_sample("Z"),
            R.equipped_for_soil_analysis("X"),
        ]
        self.add_output_action_with_derived("navigate", R.navigate_to_soil("X", "Z"), body, guard_level=NAVIGATE_GUARD)

        # move to uncollected goal rock
        body = deepcopy(navigate_priority)
        body += [
            R.uncollected_goal_rock("Z"),
            R.at_rock_sample("Z"),
            R.equipped_for_rock_analysis("X"),
        ]
        self.add_output_action_with_derived("navigate", R.navigate_to_rock("X", "Z"), body, guard_level=NAVIGATE_GUARD)

        # move to uncalibrated calibration subgoal
        body = deepcopy(navigate_priority)
        body += [
            R.uncollected_goal_image("O", "M"),
            R.supports_mode("X", "I", "M"),
            R.calibration_target("I", "T"),
            R.visible_from("T", "Z"),
        ]
        self.add_output_action_with_derived(
            "navigate", R.navigate_to_calibrate("X", "Z"), body, guard_level=NAVIGATE_GUARD
        )

        # move to uncollected goal image
        body = deepcopy(navigate_priority)
        body += [
            R.uncollected_goal_image("O", "M"),
            R.supports_mode("X", "I", "M"),
            R.calibrated("I", "X"),
            R.visible_from("O", "Z"),
        ]
        self.add_output_action_with_derived("navigate", R.navigate_to_image("X", "Z"), body, guard_level=NAVIGATE_GUARD)

        # move to communicate soil data
        body = deepcopy(navigate_priority)
        body += [
            R.ug_communicated_soil_data("P"),
            R.have_soil_analysis("X", "P"),
            R.at_lander("L", "Z1"),
            R.visible("Z", "Z1"),
        ]
        self.add_output_action_with_derived(
            "navigate", R.navigate_to_comm_soil("X", "Z"), body, guard_level=NAVIGATE_GUARD
        )

        # move to communicate rock data
        body = deepcopy(navigate_priority)
        body += [
            R.ug_communicated_rock_data("P"),
            R.have_rock_analysis("X", "P"),
            R.at_lander("L", "Z1"),
            R.visible("Z", "Z1"),
        ]
        self.add_output_action_with_derived(
            "navigate", R.navigate_to_comm_rock("X", "Z"), body, guard_level=NAVIGATE_GUARD
        )

        # move to communicate image data
        body = deepcopy(navigate_priority)
        body += [
            R.ug_communicated_image_data("O", "M"),
            R.have_image("X", "O", "M"),
            R.at_lander("L", "Z1"),
            R.visible("Z", "Z1"),
        ]
        self.add_output_action_with_derived(
            "navigate", R.navigate_to_comm_image("X", "Z"), body, guard_level=NAVIGATE_GUARD
        )

        """ drop(?x - rover ?y - store) """
        self.add_output_action("drop", [])

        """ calibrate(?r - rover ?i - camera ?t - objective ?w - waypoint) """
        body = [
            R.supports_mode("R", "I", "M"),
            R.uncollected_goal_image("O", "M"),
            ~R.derivable_calibrated_and_supports_mode("M"),
        ]
        self.add_output_action("calibrate", body)

        """ collecting """
        self.add_output_action("sample_soil", [R.uncollected_goal_soil("P")])
        self.add_output_action("sample_rock", [R.uncollected_goal_rock("P")])
        self.add_output_action("take_image", [R.uncollected_goal_image("O", "M")])

        """ communicating """
        self.add_output_action("communicate_soil_data", [R.ug_communicated_soil_data("P")])
        self.add_output_action("communicate_rock_data", [R.ug_communicated_rock_data("P")])
        self.add_output_action("communicate_image_data", [R.ug_communicated_image_data("O", "M")])
