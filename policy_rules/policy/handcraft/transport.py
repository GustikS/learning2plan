from itertools import combinations

import networkx as nx
from neuralogic.core import Aggregation, C, Combination, Metadata, R, Template, Transformation, V
from pymimir import Atom, Domain, Problem
from typing_extensions import override

from ..policy import Policy
from ..policy_learning import FasterLearningPolicy, LearningPolicy


class TransportPolicy(FasterLearningPolicy):
    def __init__(self, domain: Domain, debug=0):
        super().__init__(domain, debug)
        self._statics = {"road"}

    def setup_test_problem(self, problem: Problem):
        """ compute shortest paths with networkx """
        super().setup_test_problem(problem)
        ilg_atoms = self.get_ilg_facts(problem.initial)
        G = nx.Graph()
        for atom in ilg_atoms:
            if atom.predicate == "ap_road":
                assert len(atom.objects) == 2
                G.add_edge(atom.objects[0], atom.objects[1])
                G.add_edge(atom.objects[1], atom.objects[0])
        fw = nx.floyd_warshall(G, weight="weight")
        results = {a: dict(b) for a, b in fw.items()}
        locs = sorted(list(results.keys()))
        # print(results)

        new_atoms = []
        for a, b in combinations(locs, 2):
            dist = [results[a][b] for _ in range(self.dim)]
            atom = R.get("distance")(C.get(a), C.get(b))[dist]
            atom.fixed()
            # atom = R.get("distance")(C.get(a), C.get(b))
            new_atoms.append(atom)
        self.preprocessed_distances = new_atoms

    def _debug_inference(self):
        print("Inference for current state:")
        super()._debug_inference()
        # self._grounding_debug()
        self._debug_inference_actions()
        self._debug_inference_helper(R.derivable_pickup, newline=True)
        self._debug_inference_helper(R.derivable_drop, newline=True)
        self._debug_inference_helper(R.distance("A", "B"), newline=True)
        # self._debug_inference_helper(R.macro_drive("V", "L1", "L2"), newline=True)
        print("=" * 80)

    @override
    def _get_atoms_from_state(self, state: list[Atom]):
        ret = super()._get_atoms_from_state(state) + self.preprocessed_distances
        return ret

    @override
    def _query_macro_actions(self):
        return []

        # assignments = self._engine.query(R.macro_drive("V", "L1", "L2"))
        # param_to_index = {p.name: i for i, p in enumerate(schema.parameters)}
        # for value, assignment in assignments:
        #     objects = [None] * len(schema.parameters)
        #     for var, val in assignment.items():
        #         idx = param_to_index[f"?{var.lower()}"]
        #         obj = self._name_to_object[val]
        #         objects[idx] = obj
        #     ret_action = Action.new(self._problem, schema, objects)
        #     ret.append((value, ret_action))

    @override
    def _add_derived_predicates(self):
        self.add_rule(R.derivable_pickup, R.pickup("V", "L", "P", "S1", "S2"))
        self.add_rule(R.derivable_drop, R.drop("V", "L", "P", "S1", "S2"))

        # # drive vehicle to a package not yet at its goal to pick up
        # body = [
        #     R.ap_at("P", "L2"),
        #     R.ug_at("P", "Goal_location"),
        #     R.ap_at("V", "L1"),
        #     ~R.derivable_pickup,
        #     ~R.derivable_drop,
        #     # possibly help with the message passing?
        #     R.distance("L1", "L2"),
        #     R.distance("L1", "Goal_location"),
        #     R.distance("L2", "Goal_location"),
        # ]
        # self.add_rule(R.macro_drive("V", "L1", "L2"), body)

        # # drive vehicle to a goal location to drop off package
        # body = [
        #     R.ap_in("P", "V"),  # package in truck
        #     R.ug_at("P", "L2"),
        #     R.ap_at("V", "L1"),
        #     ~R.derivable_pickup,
        #     ~R.derivable_drop,
        #     # possibly help with the message passing?
        #     R.distance("L1", "L2"),
        # ]
        # self.add_rule(R.macro_drive("V", "L1", "L2"), body)

        # # compute shortest paths with help of road_static with non-parameterisable weights above
        # # NOTE with my implementation, Neuralogic cannot compute shortest paths correctly due to cycles
        # metadata = Metadata(
        #     aggregation=Aggregation.MIN,
        #     transformation=Transformation.IDENTITY,
        # )
        # rule1 = (R.shortest("L1", "L2") <= R.road_static("L1", "L2"))
        # # rule1.head.fixed()
        # rule2 = (R.shortest("L1", "L2") <= (R.road_static("L1", "L3"), R.shortest("L3", "L2")))
        # # rule2.head.fixed()
        # self._template += rule1 | metadata
        # self._template += rule2 | metadata

    @override
    def _add_policy_rules(self):
        """pickup(?v - vehicle ?l - location ?p - package ?s1 ?s2 - size)"""
        # pick up any package not at goal location
        body = [R.ug_at("P", "Goal_location")]
        self.add_output_action("pickup", body)

        """ drop(?v - vehicle ?l - location ?p - package ?s1 ?s2 - size) """
        # drop any package at goal location
        body = [R.ug_at("P", "L")]
        self.add_output_action("drop", body)

        """ drive(?v - vehicle ?l1 ?l2 - location) """
        # last priority for drive (check if can pick up and drop anything first)
        # note in some contrived cases, this may not be optimal due to needing to traverse over a
        # location but not needing to pick up or drop anything

        # drive vehicle to nearest goal package to pick up
        body = [
            R.ap_at("P", "Loc_of_package"),
            R.ug_at("P", "Goal_location"),
            # R.distance("L2", "Loc_of_package"),
            ~R.derivable_pickup,
            ~R.derivable_drop,
        ]
        self.add_output_action("drive", body)

        # drive vehicle to nearest goal location to drop off package
        body = [
            R.ap_in("P", "V"),  # package in truck
            R.ug_at("P", "Goal_location"),
            # R.distance("L2", "Goal_location"),
            ~R.derivable_pickup,
            ~R.derivable_drop,
        ]
        self.add_output_action("drive", body)
