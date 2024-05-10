from dataclasses import dataclass


@dataclass
class StrAtom:
    predicate: str
    objects: list[str]

    def __str__(self):
        return f"{self.predicate}({', '.join(self.objects)})"
