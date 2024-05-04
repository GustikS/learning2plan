from typing_extensions import override
from util.str_atom import StrAtom

from .derived_predicate import DerivedPredicate


class NonExistential(DerivedPredicate):
    def __init__(self):
        super().__init__("non_existential")

    @override
    def compute(self, atoms: list[StrAtom]) -> list[StrAtom]:
        raise NotImplementedError
