from abc import ABC, abstractmethod

from util.str_atom import StrAtom


class DerivedPredicate(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def compute(self, atoms: list[StrAtom]) -> list[StrAtom]:
        raise NotImplementedError
