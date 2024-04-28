from abc import ABC

import neuralogic
import jpype
import jpype.imports


# %%

class Backend:

    def __init__(self):
        try:
            neuralogic.initialize(debug_mode=False)
        except:
            pass  # already initialized

        self.domain = self.get_java("cz.cvut.fel.ida.logic.grounding.planning.Domain")
        self.state = self.get_java("cz.cvut.fel.ida.logic.grounding.planning.State")
        self.instance = self.get_java("cz.cvut.fel.ida.logic.grounding.planning.Instance")
        self.action = self.get_java("cz.cvut.fel.ida.logic.grounding.planning.Action")

        self.planner = self.get_java("cz.cvut.fel.ida.logic.grounding.planning.Planner")()
        self.matching = self.planner.matching

        self.constant = self.get_java("cz.cvut.fel.ida.logic.Constant")
        self.variable = self.get_java("cz.cvut.fel.ida.logic.Variable")
        self.predicate = self.get_java("cz.cvut.fel.ida.logic.Predicate")

        self.literal = self.get_java("cz.cvut.fel.ida.logic.Literal")
        self.clause = self.get_java("cz.cvut.fel.ida.logic.Clause")

    def get_java(self, classname):
        return jpype.JClass(classname)

    def dummy_test(self):
        utils = jpype.JClass("cz.cvut.fel.ida.utils.math.VectorUtils")
        seq = utils.sequence(2, 20, 1)
        print(seq)


def jlist(list: []):
    return jpype.java.util.ArrayList(list)


# %%
if __name__ == "__main__":
    # b = Backend()
    # b.dummy_test()
    # %%

    # init = [R.at(C.loc1), R.next(C.loc1, C.loc2), R.next(C.loc2, C.loc3), R.next(C.loc3, C.loc4)]
    # action = R.at(V.Y) <= R.next(V.X, V.Y) & R.at(V.X)
    # goal = [R.at(C.loc4)]

    b = Backend()
    loc1 = b.constant.construct("loc1", "location")
    loc2 = b.constant.construct("loc2", "location")
    loc3 = b.constant.construct("loc3", "location")
    loc4 = b.constant.construct("loc4", "location")
    car1 = b.constant.construct("car1", "car")

    X = b.variable.construct("X", "location")
    Y = b.variable.construct("Y", "location")
    C = b.variable.construct("C", "car")

    at = b.predicate("at", 2)
    next = b.predicate("next", 2)

    static_facts = [b.literal(next, False, jlist([loc1, loc2])),
                    b.literal(next, False, jlist([loc2, loc3])),
                    b.literal(next, False, jlist([loc3, loc4]))]
    init = [b.literal(at, False, jlist([car1, loc1]))]
    goal = [b.literal(at, False, jlist([car1, loc4]))]

    action = b.action("move",
                      jlist([b.literal(at, False, jlist([C, X])),
                             b.literal(next, False, jlist([X, Y]))]),
                      jlist([b.literal(at, False, jlist([C, Y]))]),
                      jlist([b.literal(at, False, jlist([C, X]))])
                      )
    actions = [action]

    instance = b.instance("instance0", jlist(static_facts), jlist(init), jlist(goal), jlist(actions))

    planner = b.planner
    solution = planner.solveGreedy(instance)

    for state in solution:
        print(state.s.toString())
