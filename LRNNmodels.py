import neuralogic
from neuralogic.core import Relation, R, V, Template, Settings, Transformation, Aggregation, Rule
from neuralogic.dataset import Dataset
from neuralogic.nn import get_evaluator
from neuralogic.optim import Adam

from planning import PlanningDataset

neuralogic.manual_seed(1)

generic_name = "relation"
label_name = "distance"

settings = Settings(chain_pruning=True, iso_value_compression=False,
                    rule_transformation=Transformation.TANH, relation_transformation=Transformation.TANH,
                    rule_aggregation=Aggregation.AVG
                    )


def get_relational_dataset(samples):
    logic_dataset = Dataset()

    for sample in samples:
        structure = sample.to_relations()
        logic_dataset.add_example(structure)
        logic_dataset.add_query(R.get("distance")[sample.state.label])

    return logic_dataset


def get_predictions_LRNN(model, built_dataset):
    predictions = []
    model.model.reset_parameters()
    model.model.test()
    # output = model.evaluator.test(built_dataset, generator=False)
    output = model.model(built_dataset)
    return output


class GNN:
    template: Template
    model: object

    def __init__(self, samples, dim=16):
        self.dim = dim

        self.num_node_features = len(samples[0].node_feature_names())
        self.num_edge_features = len(samples[0].edge_feature_names())

        self.template = Template()

        self.template.add_rules(self.get_rules())
        self.model = self.template.build(settings)

    def get_rules(self) -> [Rule]:
        rules = []

        # A classic message passing over the edges (preprocessed binary relations)
        rules.append(R.embedding(V.X)[self.dim, self.dim] <= (
            R.get("edge")(V.Y, V.X)[self.dim, self.num_edge_features],
            R.get("node")(V.Y)[self.dim, self.num_node_features]))

        # Global pooling/readout
        rules.append(R.get(label_name)[1, self.dim] <= R.embedding(V.X))

        # # Aggregate also the zero-order predicate(s)
        # rules.append(R.get(label_name)[1, len(dataset.domain.arities[0])] <= R.get("proposition"))

        # ...and the unary predicate(s) on their own
        rules.append(R.get(label_name)[1,] <= R.get("node")(V.X)[1, self.num_node_features])

        rules.append(R.get(label_name) / 0 | [
            Transformation.IDENTITY])  # we will want to end up with Identity (or Relu) as this is a distance regression task

        return rules
