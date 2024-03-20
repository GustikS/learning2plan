import neuralogic
from neuralogic.core import Relation, R, V, Template, Settings, Transformation, Aggregation, Rule
from neuralogic.dataset import Dataset
from neuralogic.nn import get_evaluator
from neuralogic.nn.module import GCNConv
from neuralogic.optim import Adam
from typing_extensions import deprecated

neuralogic.manual_seed(1)

generic_name = "relation"
label_name = "distance"


def get_relational_dataset(samples):
    logic_dataset = Dataset()

    for sample in samples:
        structure = sample.to_relations()
        logic_dataset.add_example(structure)
        logic_dataset.add_query(R.get(label_name)[sample.state.label])

    return logic_dataset


def get_predictions_LRNN(model, built_dataset, reset_weights=True):
    predictions = []
    if reset_weights:
        model.model.reset_parameters()
    model.model.test()
    # output = model.evaluator.test(built_dataset, generator=False)
    output = model.model(built_dataset)
    return output


class LRNN:
    template: Template
    model: object

    def __init__(self, samples, model_class=GCNConv, num_layers=3, hidden_channels=8, aggr="add"):
        sample = samples[0]
        if sample:
            first_node_features = next(iter(sample.node_features.items()))[1]
            self.num_node_features = len(first_node_features)
            try:
                self.num_edge_features = len(sample.edge_features[0])
            except:
                num_edge_features = -1
                for sam in samples:
                    curr = max(sam.edge_features) + 1
                    if curr > num_edge_features:
                        num_edge_features = curr
        else:
            self.num_node_features = -1
            self.num_edge_features = -1

        self.model_class = model_class
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.aggregation = self.get_aggregation(aggr)

        self.settings = Settings(chain_pruning=True, iso_value_compression=False,
                                 rule_transformation=Transformation.TANH, relation_transformation=Transformation.TANH,
                                 rule_aggregation=self.aggregation)

        self.template = Template()
        self.load_gnn_template(self.template)
        self.model = self.template.build(self.settings)

    def get_aggregation(self, aggr):
        if aggr == "add":
            return Aggregation.SUM
        elif aggr == "mean":
            return Aggregation.AVG
        elif aggr == "max":
            return Aggregation.MAX

    def load_gnn_template(self, template: Template):
        template.add_module(
            self.model_class(in_channels=self.num_node_features, out_channels=self.hidden_channels, output_name="h0",
                             feature_name="node", edge_name="edge")
        )
        for l in range(1, self.num_layers):
            template.add_module(
                self.model_class(in_channels=self.hidden_channels, out_channels=self.hidden_channels,
                                 output_name=f"h{l}", feature_name=f"h{l - 1}", edge_name="edge")
            )
        template += R.get(label_name)[1, self.hidden_channels] <= R.get(f"h{self.num_layers}")(V.X)
        # todo add this?
        template += R.get(label_name)[1,] <= R.get("node")(V.X)[1, self.num_node_features]

        template += R.get(label_name) / 0 | [Transformation.IDENTITY]

    @deprecated
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
