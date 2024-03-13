from abc import ABC

import neuralogic
import jpype

neuralogic.initialize()

# %% neuralogic call test
utils = jpype.JClass("cz.cvut.fel.ida.utils.math.VectorUtils")
seq = utils.sequence(2, 20, 1)
print(seq)


# %%

class Java(ABC):
    backend: str

    def __init__(self):
        self.java = jpype.JClass(self.backend)

class Domain(Java):
    backend = "cz.cvut.fel.ida.logic.grounding.planning.Domain"


class State(Java):
    backend = "cz.cvut.fel.ida.logic.grounding.planning.State"


class Instance(Java):
    backend = "cz.cvut.fel.ida.logic.grounding.planning.Instance"


class Action(Java):
    backend = "cz.cvut.fel.ida.logic.grounding.planning.Instance"


class Planner(Java):
    backend = "cz.cvut.fel.ida.logic.grounding.planning.Planner"

# %%

s = State().java(None)
print(s.getId())

p = Planner().java()
print(p.matching)
