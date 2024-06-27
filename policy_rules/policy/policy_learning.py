import pickle
from typing_extensions import override

from neuralogic.core import Template, Settings, R, C, Transformation, Aggregation
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.inference import EvaluationInferenceEngine
from pymimir import Domain, Problem, Atom, Action

from modelling.samples import get_filename
from modelling.testing import load_model
from modelling.training import build_samples, store_weights, store_template
from policy_rules.policy.policy import Policy, Schema


class LearningPolicy(Policy):

    def __init__(self, domain: Domain, template_path: str, debug=0, train: bool = True):
        # TODO remove the Problem dependency - a policy should generalize over different problems, right?
        super().__init__(domain, template_path, train, debug)

        # we can setup all the learning/evaluation-related settings here
        self.settings = Settings(
            iso_value_compression=False,
            chain_pruning=False,
            rule_transformation=Transformation.RELU,
            rule_aggregation=Aggregation.SUM,  # avg for better generalization then
            relation_transformation=Transformation.RELU,
            epochs=100
        )

        # no need to recreate the template with every new state, we can retain it
        try:
            self._template = load_model(template_path)
        except FileNotFoundError:
            print(f"No stored template found at {template_path} - will train one and store it there instead!")
            self._init_template()
            if train:
                training_data_path = get_filename(domain.name, False, 'lrnn', "..", "")
                try:
                    LearningPolicy.train_parameters(self._template.build(self.settings),
                                                    training_data_path,
                                                    save_model_path=template_path)
                except Exception as e:
                    print(f"No training possible (no data available?) from: {training_data_path}")
                    print(e)
            else:
                print("Resorting to a pure handcrafted template")

        # an inference engine just like before, but this one returns the numeric values too
        self._engine = EvaluationInferenceEngine(self._template, self.settings)
        self.model = self._engine.model

        self.header2query = {}
        for schema in self._schemata:
            query = self.relation_from_schema(schema)
            self.header2query[query.predicate.name] = query

    @override
    def solve(self, state: list[Atom]) -> list[Action]:
        ilg_atoms = self.get_ilg_facts(state)
        lrnn_atoms = [R.get(atom.predicate)([C.get(obj) for obj in atom.objects]) for atom in ilg_atoms]
        self._engine.set_knowledge(lrnn_atoms)
        return self.query_actions()

    @override
    def get_action_substitutions(self, action_name):
        action_header = self.header2query[action_name]  # todo store these
        # there is a little bug/discrepancy between the eval/inference in neuralogic
        # so we need to reformat a bit here (upper string instead of capital var) - will correct that in next version
        for term in action_header.terms:
            term.name = term.name.upper()
        substitutions = list(self._engine.query(action_header))
        assignments = []
        for val, subs in substitutions:
            subss = {}
            for var, const in subs.items():
                subss[var.name.capitalize()] = const
            assignments.append((val, subss))
        return assignments

    @override
    def _get_schema_preconditions(self, schema: Schema, dim=1, num_layers=1) -> list[BaseRelation]:
        if isinstance(schema, str):
            schema = self._name_to_schema[schema]
        body = super().get_schema_preconditions(schema)
        variables = [p.name.replace("?", "").upper() for p in schema.parameters]
        # just extend each action with the latent representation of each involved object
        # todo next - merge with the message passing here...
        body += [R.get(f'h_{num_layers}')(var)[dim, dim] for var in variables]
        return body

    @staticmethod
    def train_parameters(model, lrnn_dataset_path: str, epochs: int = 100, save_model_path: str = None):
        neural_samples, logic_samples = build_samples(model, lrnn_dataset_path)
        results = model(neural_samples, train=True, epochs=epochs)
        print(results)
        if save_model_path:
            store_weights(model, save_model_path)
            store_template(model, save_model_path)

    def reset_parameters(self):
        self.model.reset_parameters()

    def load_parameters(self, file_path: str = None):
        with open(file_path + "_weights", 'rb') as f:
            weights = pickle.load(f)
            self.model.load_state_dict(weights)


class FasterEvaluationPolicy(LearningPolicy):

    def __init__(self, domain: Domain, template_path: str, debug=0, train: bool = True):
        super().__init__(domain, template_path, debug, train)


    @override
    def solve(self, state: list[Atom]) -> list[Action]:
        pass