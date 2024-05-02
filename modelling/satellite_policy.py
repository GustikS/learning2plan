import argparse
import time
from pathlib import Path
from pprint import pprint

import pymimir
from neuralogic.core import C, R, Template, V
from neuralogic.inference.inference_engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_path", type=Path, help="Path")
    args = parser.parse_args()
    domain_path = "l4np/satellite/classic/domain.pddl"
    assert Path(domain_path).exists(), f"Domain file not found: {domain_path}"
    time_start = time.time()
    domain = pymimir.DomainParser(str(domain_path)).parse()
    problem = pymimir.ProblemParser(str(args.problem_path)).parse(domain)
    successor_generator = pymimir.LiftedSuccessorGenerator(problem)
    initial_state = problem.create_state(problem.initial)
    schemata = domain.action_schemas
    time_end = time.time()
    print(f"Parse time: [{time_end - time_start:.1f} seconds]")

    template = Template()

    ## schemata
    for schema in schemata:
        parameters = [p.name for p in schema.parameters]
        parameter_remap = {p: f"X{i}" for i, p in enumerate(parameters)}

        head_vars = [parameter_remap[p] for p in parameters]
        head = R.get(schema.name)(head_vars)

        body = []
        for p in schema.precondition:
            toks = p.atom.get_name().replace(" ", "").split("(")
            assert len(toks) <= 2
            predicate = toks[0]
            objects = toks[1].replace(")", "").split(",")
            prec_vars = [V.get(parameter_remap[a]) for a in objects]
            literal = R.get(predicate)(prec_vars)
            if p.negated:
                literal = ~literal
            body.append(literal)

        body = tuple(body)
        template.add_rule(head <= body)

    ## state
    for atom in initial_state.get_atoms():
        toks = atom.get_name().split("(")
        assert len(toks) <= 2
        predicate = toks[0]
        objects = toks[1].replace(")", "").replace(" ", "").split(",")
        objects = [C.get(obj) for obj in objects]
        fact = R.get(predicate)(objects)
        template += [fact]

    ## debug
    print("="*80)
    print("Template:")
    print(template)

    ## solve
    print("="*80)
    engine = InferenceEngine(template)
    turn_to_actions = engine.query(R.turn_to(V.X0, V.X1, V.X2))
    switch_on_actions = engine.query(R.switch_on(V.X0, V.X1))
    print(f"{turn_to_actions=} <==== this should NOT be empty...")
    print(f"{switch_on_actions=}")

    ## successor generator usage
    applicable_actions = successor_generator.get_applicable_actions(initial_state)
    applicable_actions = [a.get_name() for a in applicable_actions]
    print(f'actions computed by mimir=')
    pprint(applicable_actions)
    print("="*80)
    # succ = applicable_actions[0].apply(initial_state)
    # print(len(applicable_actions))


if __name__ == "__main__":
    main()
