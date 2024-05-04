from dataclasses import dataclass


@dataclass
class StrAtom:
    predicate: str
    objects: list[str]