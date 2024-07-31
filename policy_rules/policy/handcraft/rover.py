from copy import deepcopy
from itertools import combinations, product

import networkx as nx
from neuralogic.core import Aggregation, C, Combination, Metadata, R, Template, Transformation, V
from pymimir import Atom, Domain, Problem
from typing_extensions import override

from ..policy import Policy
from ..policy_learning import FasterLearningPolicy, LearningPolicy

DERV_COMM_GUARD = 3
UNCOLLECTED_GUARD = 6
DERV_UNCOLLECTED_GUARD = 9
ACTION_GUARD = 12

_DEBUG_NAVIGATE = True


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

        print("=" * 80)
        for relation in [
            R.collected_rock("P"),
            R.uncollected_goal_soil("P"),
            R.collected_soil("P"),
            R.uncollected_goal_rock("P"),
            R.collected_image("O", "M"),
            R.uncollected_goal_image("O", "M"),
        ]:
            self._debug_inference_helper(relation, newline=True)

        print("=" * 80)
        self._debug_inference_helper(R.supports_mode("R", "I", "M"), newline=True)

        # print("=" * 80)
        # for relation in [
        #     R.derive_communicate_soil_data,
        #     R.derive_communicate_rock_data,
        #     R.derive_communicate_image_data,
        # ]:
        #     self._debug_inference_helper(relation, newline=True)

        if _DEBUG_NAVIGATE:
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
        self.add_rule(R.collected_rock("P"), [R.have_rock_analysis("R", "P")])
        self.add_rule(R.collected_soil("P"), [R.have_soil_analysis("R", "P")])
        self.add_rule(R.collected_image("O", "M"), [R.have_image("R", "O", "M")])

        self.add_rule(
            R.uncollected_goal_rock("P"),
            [R.ug_communicated_rock_data("P"), ~R.collected_rock("P")],
            guard_level=UNCOLLECTED_GUARD,
        )

        self.add_rule(
            R.uncollected_goal_soil("P"),
            [R.ug_communicated_soil_data("P"), ~R.collected_soil("P")],
            guard_level=UNCOLLECTED_GUARD,
        )

        self.add_rule(
            R.uncollected_goal_image("O", "M"),
            [R.ug_communicated_image_data("O", "M"), ~R.collected_image("O", "M")],
            guard_level=UNCOLLECTED_GUARD,
        )

        self.add_rule(
            R.supports_mode("R", "I", "M"),
            [R.on_board("I", "R"), R.equipped_for_imaging("R"), R.supports("I", "M")],
        )

        # exists predicate
        for tup in [
            ("empty_soil", [R.empty("S"), R.equipped_for_soil_analysis("S")]),
            ("empty_rock", [R.empty("S"), R.equipped_for_rock_analysis("S")]),
            ("sample_soil", R.sample_soil("X", "S", "P")),
            ("sample_rock", R.sample_rock("X", "S", "P")),
            ("drop", R.drop("X", "Y")),
            ("calibrate", R.calibrate("R", "I", "T", "W")),
            ("take_image", R.take_image("R", "P", "O", "I", "M")),
            ("comm_soil_data", R.communicate_soil_data("R", "L", "P", "X", "Y"), DERV_COMM_GUARD),
            ("comm_rock_data", R.communicate_rock_data("R", "L", "P", "X", "Y"), DERV_COMM_GUARD),
            ("comm_image_data", R.communicate_image_data("R", "L", "O", "M", "X", "Y"), DERV_COMM_GUARD),
            ("uncollected_soil", R.uncollected_goal_soil("P"), DERV_UNCOLLECTED_GUARD),
            ("uncollected_rock", R.uncollected_goal_rock("P"), DERV_UNCOLLECTED_GUARD),
            ("uncollected_image", R.uncollected_goal_image("O", "M"), DERV_UNCOLLECTED_GUARD),
        ]:
            if len(tup) == 2:
                head, body = tup
                guard_level = -1
            else:
                assert len(tup) == 3
                head, body, guard_level = tup
            if not isinstance(body, list):
                body = [body]
            head = R.get(f"derv_{head}")
            self.add_rule(head, body, guard_level=guard_level)

        self.add_rule(
            R.derv_calibrated_and_supports_mode("M"), [R.calibrated("I", "R"), R.supports_mode("R", "I", "M")]
        )


    def _add_navigate_rule(self, body, description):
        """ Helper for adding and debugging navigate rules """

        navigate_priority = [
            # Tne domain misses precondition that we don't go to the same place.
            ~R.at("X", "Z"),
            # The following prioritises other actions over navigating
            ~R.derv_sample_soil,
            ~R.derv_sample_rock,
            ~R.derv_drop,
            ~R.derv_calibrate,
            ~R.derv_take_image,
            ~R.derv_comm_soil_data,
            ~R.derv_comm_rock_data,
            ~R.derv_comm_image_data,
        ]

        extended_body = body + deepcopy(navigate_priority)  # Python being annoying with copying
        self.add_output_action("navigate", extended_body, guard_level=ACTION_GUARD)

        if _DEBUG_NAVIGATE:
            self.add_rule(R.get(f"navigate_to_{description}")("X", "Z"), extended_body, guard_level=ACTION_GUARD)


    @override
    def _add_policy_rules(self):
        # https://ipc2023-learning.github.io/talk.pdf
        # 1. for each rock/soil data in the goal, get a rover equipped for
        # rock/soil analysis and can move to that waypoint, sample and drop it

        """navigate(?x - rover ?y - waypoint ?z - waypoint)"""

        # move to uncollected goal soil
        body = [
            R.uncollected_goal_soil("Z"),
            R.at_soil_sample("Z"),
            R.equipped_for_soil_analysis("X"),
        ]
        self._add_navigate_rule(body, "soil")

        # move to uncollected goal rock
        body = [
            R.uncollected_goal_rock("Z"),
            R.at_rock_sample("Z"),
            R.equipped_for_rock_analysis("X"),
        ]
        self._add_navigate_rule(body, "rock")

        # move to uncalibrated calibration subgoal
        body = [
            R.uncollected_goal_image("O", "M"),
            R.supports_mode("X", "I", "M"),
            R.calibration_target("I", "T"),
            R.visible_from("T", "Z"),
            ~R.derv_calibrated_and_supports_mode("M"),
        ]
        self._add_navigate_rule(body, "calibrate")

        # move to uncollected goal image
        body = [
            R.uncollected_goal_image("O", "M"),
            R.supports_mode("X", "I", "M"),
            R.calibrated("I", "X"),
            R.visible_from("O", "Z"),
        ]
        self._add_navigate_rule(body, "image")

        # move to communicate soil data
        body = [
            R.ug_communicated_soil_data("P"),
            R.have_soil_analysis("X", "P"),
            R.at_lander("L", "Z1"),
            R.visible("Z", "Z1"),
        ]
        self._add_navigate_rule(body, "comm_soil")

        # move to communicate rock data
        body = [
            R.ug_communicated_rock_data("P"),
            R.have_rock_analysis("X", "P"),
            R.at_lander("L", "Z1"),
            R.visible("Z", "Z1"),
        ]
        self._add_navigate_rule(body, "comm_rock")

        # move to communicate image data
        body = [
            R.ug_communicated_image_data("O", "M"),
            R.have_image("X", "O", "M"),
            R.at_lander("L", "Z1"),
            R.visible("Z", "Z1"),
        ]
        self._add_navigate_rule(body, "comm_image")

        """ drop(?x - rover ?y - store) """
        self.add_output_action("drop", [R.derv_uncollected_soil, ~R.derv_empty_soil], guard_level=ACTION_GUARD)
        self.add_output_action("drop", [R.derv_uncollected_rock, ~R.derv_empty_rock], guard_level=ACTION_GUARD)

        """ calibrate(?r - rover ?i - camera ?t - objective ?w - waypoint) """
        body = [
            R.supports_mode("R", "I", "M"),
            R.uncollected_goal_image("O", "M"),
            ~R.calibrated("I", "R"),
        ]
        self.add_output_action("calibrate", body, guard_level=ACTION_GUARD)

        """ collecting """
        self.add_output_action("sample_soil", [R.uncollected_goal_soil("P")], guard_level=ACTION_GUARD)
        self.add_output_action("sample_rock", [R.uncollected_goal_rock("P")], guard_level=ACTION_GUARD)
        self.add_output_action("take_image", [R.uncollected_goal_image("O", "M")], guard_level=ACTION_GUARD)

        """ communicating """
        self.add_output_action("communicate_soil_data", [R.ug_communicated_soil_data("P")])
        self.add_output_action("communicate_rock_data", [R.ug_communicated_rock_data("P")])
        self.add_output_action("communicate_image_data", [R.ug_communicated_image_data("O", "M")])
