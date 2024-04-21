import neuralogic
from neuralogic.core import Relation, R, V, Template, Settings, Transformation, Aggregation, Rule
from neuralogic.dataset import Dataset
from neuralogic.nn import get_evaluator
from neuralogic.nn.loss import MSE
from neuralogic.nn.module import GCNConv
from neuralogic.optim import Adam, SGD
from typing_extensions import deprecated

from learning2plan.logic import LogicLanguage
from learning2plan.planning import PlanningDataset

neuralogic.manual_seed(1)

generic_name = "relation"
label_name = "distance"


def get_relational_dataset(samples, include_raw_relations=True):
    logic_dataset = Dataset()

    for sample in samples:
        structure = sample.to_relations()
        if include_raw_relations:
            structure.extend(sample.raw_relations())
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


def get_trained_model_lrnn(dataset: PlanningDataset, encoding, model_type,
                           optimizer="ADAM", learning_rate=0.001, epochs=100,
                           batch_size=1, include_actions=True):
    samples = [state.get_sample(encoding) for state in dataset.states]
    actions = dataset.actions if include_actions else None
    model = LRNN(samples, actions=actions, model_class=model_type, num_layers=1, hidden_channels=8, aggr="add")
    model.settings.optimizer = Adam(learning_rate) if optimizer == "ADAM" else SGD(learning_rate)
    model.settings.learning_rate = learning_rate
    model.settings.epochs = epochs
    model.train(samples, batch_size=batch_size)
    return model


class LRNN:
    template: Template
    model: object

    def __init__(self, samples, actions=None, model_class=GCNConv, num_layers=3, hidden_channels=8, aggr="add"):
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
                                 rule_transformation=Transformation.RELU, relation_transformation=Transformation.RELU,
                                 rule_aggregation=self.aggregation, error_function=MSE())

        self.template = Template()
        self.load_gnn_template(self.template)
        self.load_actions_template(self.template, actions)
        self.template.draw(filename="./template_img.png")
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
        template += R.get(label_name)[1, self.hidden_channels] <= R.get(f"h{self.num_layers - 1}")(V.X)
        # todo add this?
        template += R.get(label_name)[1,] <= R.get("node")(V.X)[1, self.num_node_features]

        template += R.get(label_name) / 0 | [Transformation.IDENTITY]

    def load_actions_template(self, template, actions):
        for action in actions:
            all_terms = set()
            preconditions = []
            for prec in action.preconditions:
                all_terms.update(prec.terms)
                preconditions.append(R.get(prec.predicate.name)(prec.terms)[self.hidden_channels, 1])
            template += R.get(action.name)(all_terms) <= preconditions
            template += R.get(label_name)[1, self.hidden_channels] <= R.get(action.name)(all_terms)[
                self.hidden_channels, self.hidden_channels]

    def train(self, samples, batch_size=1):
        evaluator = get_evaluator(self.template, self.settings)
        relational_dataset = get_relational_dataset(samples)
        built_dataset = evaluator.build_dataset(relational_dataset, batch_size=batch_size)
        for sample in built_dataset:
            sample.draw(filename="sample.png")
            break
        for i, (current_total_loss, number_of_samples) in enumerate(evaluator.train(built_dataset.samples)):
            print(i, ": ", current_total_loss)
        # todo check trained model retains weights

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
