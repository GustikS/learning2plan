# Description of json Objects
Each json object has the `@dataclass`-like structure

```
predicates: Dict[str, int]
functions: Dict[str, int]
schemata: Dict[str, int]
problems: List[Problem]
```

- dictionaries map the keys to their arities
- `Problem` has the structure

```
problem_pddl: str
objects: List[str]
static_facts: List[str]
static_fluents: Dict[str, float]
boolean_goals: List[str]
numeric_goals: List[str]
states: List[State]
```

- `State` has the structure

```
facts: List[str]
fluents: Dict[str, float]
h: Optional[float]
optimal_action: Optional[str]
parent_facts: Optional[List[str]]
parent_fluents: Optional[List[str]]
```

- `h` is the optimal cost to go and `optimal_action` is the optimal action of the state in the plan trace for states in the plan trace, and None otherwise. Note that there may exist more than one optimal action.
- all states except the initial state will have a parent state described by `parent_facts` and `parent_fluents` and this parent state is in the plan trace. Note that parents are also not unique, and we only chose them from the plan trace.
