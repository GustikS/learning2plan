import seaborn
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from neuralogic.core import Template, R, V, Transformation, Settings
from neuralogic.dataset import FileDataset
from neuralogic.nn import get_evaluator
from neuralogic.nn.loss import MSE, CrossEntropy

from neuralogic.nn.module import GCNConv, SAGEConv, GINConv, GATv2Conv
from neuralogic.optim import Adam
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from learning2plan.modelling.samples import load_json, parse_domain, export_problems


def basic_regression_template(predicates, dim=10):
    template = Template()

    template += anonymous_predicates(predicates, dim)
    template += object_info_aggregation(max(predicates.values()), dim)
    template += gaifman_edges(max(predicates.values()), dim, "edge")
    # template += custom_message_passing("edge", "h0", dim)
    template += gnn_message_passing(f"edge", dim)
    template += final_pooling(dim, layers=[0, 1, 2])

    return template


def basic_classification_template(predicates, dim=10):
    template = Template()
    # TODO the action prediction...


def final_pooling(hidden, query_name="distance", layers=[0, 1, 2]):
    rules = []
    for layer in layers:
        rules.append(R.get(query_name)[1, hidden] <= R.get(f"h{layer}")(V.X)[hidden, hidden])
    return rules


def object_info_aggregation(max_arity, dim, unary_only=False):
    rules = []
    max_arity = 1 if unary_only else max_arity
    for arity in range(0, max_arity + 1):
        variables = [f"X{ar}" for ar in range(arity)]
        # all objects calculate their embeddings by aggregating info from connected atoms
        positions = range(arity) if arity else [0]  # add also nullary relations here...
        rules += [R.get(f"h{0}")(f'X{i}')[dim, dim] <= R.get(f"{arity}-ary")(variables)[dim, dim] for i in positions]
    return rules


def anonymous_predicates(predicates, dim):
    rules = []
    for predicate, arity in predicates.items():  # anonymizing/embedding all domain predicates
        variables = [f"X{ar}" for ar in range(arity)]
        rules.append(R.get(f"{arity}-ary")(variables)[dim, dim] <= R.get(f"{predicate}")(variables)[dim,])
    return rules


def gaifman_edges(max_arity, dim, edge_name="edge"):
    rules = []
    for arity in range(0, max_arity + 1):
        variables = [f"X{ar}" for ar in range(arity)]
        pairs = ((i, j) for i in variables for j in variables if i != j)  # all pairwise interactions
        rules += [R.get(edge_name)(pair)[dim, dim] <= R.get(f"{arity}-ary")(variables)[dim, dim] for pair in pairs]
    return rules


def custom_message_passing(binary_relation, unary_relation, dim, bidirectional=True):
    rules = []
    rules.append(R.get("h1")(V.X)[dim, dim] <=
                 R.get(binary_relation)(V.X, V.Y)[dim, dim] & R.get(unary_relation)(V.Y)[dim, dim])
    if bidirectional:
        rules.append(R.get("h1")(V.X)[dim, dim] <=
                     R.get(binary_relation)(V.Y, V.X)[dim, dim] & R.get(unary_relation)(V.Y)[dim, dim])
    return rules


def gnn_message_passing(binary_relation, dim, num_layers=3, model_class=SAGEConv):
    rules = []
    for layer in range(1, num_layers):
        rules += model_class(dim, dim, output_name=f"h{layer}", feature_name=f"h{layer - 1}",
                             edge_name=binary_relation)()
    return rules


def build_model(data_path, template, drawing=True, regression=True):
    settings = Settings(iso_value_compression=not drawing,
                        rule_transformation=Transformation.LEAKY_RELU if regression else Transformation.TANH,
                        relation_transformation=Transformation.LEAKY_RELU if regression else Transformation.SIGMOID)
    dataset = FileDataset(f'{data_path}/examples.txt', f'{data_path}/queries.txt')
    built_samples = template.build(settings).build_dataset(dataset)
    return built_samples


def train_model(built_samples, template, regression=True):
    settings = Settings(optimizer=Adam(lr=0.001),
                        epochs=100,
                        error_function=MSE() if regression else CrossEntropy())
    evaluator = get_evaluator(template, settings)

    for i, (current_total_loss, number_of_samples) in enumerate(evaluator.train(built_samples.samples)):
        print(f'epoch: {i} total loss: {current_total_loss} samples updated: {number_of_samples}')

    target_labels, predicted_labels = [], []
    for sample, prediction in zip(built_samples.samples, evaluator.test(built_samples)):
        print(f"Target: {sample.target}, Predicted: {round(prediction)} ({prediction})")
        target_labels.append(sample.target), predicted_labels.append(round(prediction))

    return target_labels, predicted_labels


def plot_predictions(target_labels, predicted_labels):
    data = confusion_matrix(target_labels, predicted_labels)
    figure(figsize=(20, 20))
    ax = seaborn.heatmap(data, annot=True, square=True, cmap='Blues', annot_kws={"size": 7}, cbar_kws={"shrink": 0.5})
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.savefig('confusion.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    domain = "blocksworld"
    # domain = "satellite"
    json_data = load_json(domain, numeric=False)
    problems, predicates, actions = parse_domain(json_data)
    data_path = export_problems(problems, domain)

    template = basic_regression_template(predicates, dim=1)
    # template.draw("template.png")

    built_samples = build_model(data_path, template)
    # built_samples[0].draw("sample.png")

    target_labels, predicted_labels = train_model(built_samples, template)
    plot_predictions(target_labels, predicted_labels)
