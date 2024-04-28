import logging

import seaborn
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from neuralogic.core import R, Settings, Template, Transformation, V
from neuralogic.dataset import FileDataset
from neuralogic.nn import get_evaluator
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.nn.module import GATv2Conv, GCNConv, GINConv, SAGEConv
from neuralogic.optim import Adam
from samples import export_problems, load_json, parse_domain
from sklearn.metrics import confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(filename)s:%(lineno)s] %(message)s",
)

def satellite_regression_template(predicates, dim=10):
    template = Template()

    num_layers = 3
    max_arity = max(predicates.values())

    # anonymizing/embedding all domain predicates
    for predicate, arity in predicates.items():  
        variables = [f"X{ar}" for ar in range(arity)]
        template += (R.get(f"{arity}-ary_{0}")(variables)[dim, dim] <= R.get(f"{predicate}")(variables)[dim,])

    for layer in range(num_layers):
        template += object_info_aggregation(max_arity, dim, layer)
    for layer in range(num_layers - 1):
        template += atom_info_aggregation(max_arity, dim, layer)

    for layer in range(1, num_layers + 1):  # e.g. just aggregate object embeddings from the layers...
        template +=(R.get("distance")[1, dim] <= R.get(f"h_{layer}")(V.X)[dim, dim])

    return template


def basic_regression_template(predicates, dim=10):
    template = Template()

    template += anonymous_predicates(predicates, dim)

    # template += object_info_aggregation(max(predicates.values()), dim)
    # template += atom_info_aggregation(max(predicates.values()), dim)

    # template += object2object_edges(max(predicates.values()), dim, "edge")

    # template += custom_message_passing("edge", "h0", dim)
    # template += gnn_message_passing("edge", dim, num_layers=3)
    # template += gnn_message_passing(f"{2}-ary", dim)

    # template += objects2atoms_exhaustive_messages(predicates, dim)
    template += objects2atoms_anonymized_messages(max(predicates.values()), dim, num_layers=3)

    template += final_pooling(dim, layers=[1, 2, 3])

    return template


def basic_classification_template(predicates, dim=10):
    template = Template()
    # TODO the action prediction...


def anonymous_predicates(predicates, dim, input_dim=1):
    """
    map all the domain predicates to newly derived ones (anonymous) while respecting the same arity
    *input_dim* = 3 for our numeric encoding of the goal info into the predicates, or just 1 otherwise
    """
    rules = []
    for predicate, arity in predicates.items():  # anonymizing/embedding all domain predicates
        variables = [f"X{ar}" for ar in range(arity)]
        rules.append(R.get(f"{arity}-ary_{0}")(variables)[dim, dim] <= R.get(f"{predicate}")(variables)[dim, input_dim])
    return rules


def final_pooling(hidden, layers, query_name="distance"):
    """aggregate all relevant info from the computation graph for a final output"""
    rules = []
    for layer in layers:  # e.g. just aggregate object embeddings from the layers...
        rules.append(R.get(query_name)[1, hidden] <= R.get(f"h_{layer}")(V.X)[hidden, hidden])
    return rules


def object_info_aggregation(max_arity, dim, layer=0, unary_only=False, add_nullary=True):
    """objects aggregate info from all the atoms they are associated with"""
    rules = []
    max_arity = 1 if unary_only else max_arity  # only absorb unary predicates (typical "features")
    for arity in range(0, max_arity + 1):
        variables = [f"X{ar}" for ar in range(arity)]
        # all objects calculate their embeddings by aggregating info from all associated atoms
        positions = range(arity) if arity else [0] if add_nullary else []  # optionally add also nullary atoms here
        rules += [R.get(f"h_{layer + 1}")(f'X{i}')[dim, dim] <=
                  R.get(f"{arity}-ary_{layer}")(variables)[dim, dim] for i in positions]
    return rules


def atom_info_aggregation(max_arity, dim, layer=0):
    """vice-versa, atoms aggregate info from all the objects they contain"""
    rules = []
    for arity in range(0, max_arity + 1):
        variables = [f"X{ar}" for ar in range(arity)]
        rules.append(R.get(f"{arity}-ary_{layer + 1}")(variables)[dim, dim] <=
                     [R.get(f'h_{layer + 1}')(f'X{i}')[dim, dim] for i in range(arity)] + [
                         R.get(f"{arity}-ary_{layer}")(variables)[dim, dim]])
    return rules


def object2object_edges(max_arity, dim, edge_name="edge"):
    """i.e. constructing the GAIFMAN graph's binary relation (derived/anonymous)"""
    rules = []
    for arity in range(0, max_arity + 1):
        variables = [f"X{ar}" for ar in range(arity)]
        pairs = ((i, j) for i in variables for j in variables if i != j)  # all pairwise interactions
        rules += [R.get(edge_name)(pair)[dim, dim] <= R.get(f"{arity}-ary_{0}")(variables)[dim, dim] for pair in pairs]
    return rules


def objects2atoms_anonymized_messages(max_arity, dim, num_layers=3):
    """i.e. something like message-passing on the bipartite object-atom (ILG,munin,...) graph representation,
    while using the derived (anonymous) relations"""
    rules = []
    for layer in range(num_layers):
        rules += object_info_aggregation(max_arity, dim, layer)
    for layer in range(num_layers - 1):
        rules += atom_info_aggregation(max_arity, dim, layer)
    return rules


def objects2atoms_exhaustive_messages(predicates, dim, num_layers=3, object_name="h"):
    """i.e. even closer to something like GNNs on the bipartite (ILG,munin,...) graph representation,
    passing messages on the ORIGINAL relations (as opposed to the anonymized ones, which is more compact)"""
    rules = []
    for predicate, arity in predicates.items():  # anonymizing/embedding all domain predicates
        if not arity:
            continue  # here we just skip the nullary atoms
        variables = [f"X{i}" for i in range(arity)]
        for layer in range(1, num_layers):
            # objects -> atom
            rules.append(R.get(f"h_{predicate}_{layer}")(variables)[dim, dim] <=
                         [R.get(f'{object_name}_{layer}')(f'X{i}')[dim, dim] for i in range(arity)] + [
                             R.get(f"_{predicate}")(variables)])
            # atom => objects
            rules += [R.get(f'{object_name}_{layer}')(f'X{i}')[dim, dim] <=
                      R.get(f"h_{predicate}_{layer}")(variables)[dim, dim] for i in range(arity)]
    return rules


def atom2atom_messages(max_arity, dim, num_layers=3):
    # TODO the last remaining (classic) message-passing mode...
    pass


def custom_message_passing(binary_relation, unary_relation, dim, layer=1, bidirectional=True):
    """just a custom rule for passing a message/features (unary_relation) along a given binary relation (binary_relation)"""
    rules = []
    rules.append(R.get(f"h{layer}")(V.X)[dim, dim] <=
                 R.get(binary_relation)(V.X, V.Y)[dim, dim] & R.get(unary_relation)(V.Y)[dim, dim])
    if bidirectional:
        rules.append(R.get(f"h{layer}")(V.X)[dim, dim] <=
                     R.get(binary_relation)(V.Y, V.X)[dim, dim] & R.get(unary_relation)(V.Y)[dim, dim])
    return rules


def gnn_message_passing(binary_relation, dim, num_layers=3, model_class=SAGEConv):
    """classic message passing reusing some existing GNN models as implemented in LRNN rules..."""
    rules = []
    for layer in range(1, num_layers):
        rules += model_class(dim, dim, output_name=f"h_{layer + 1}", feature_name=f"h_{layer}",
                             edge_name=binary_relation)()
    return rules


def build_model(data_path, template, drawing=True, regression=True):
    logging.log(logging.INFO, "building model")
    settings = Settings(iso_value_compression=not drawing,
                        rule_transformation=Transformation.LEAKY_RELU if regression else Transformation.TANH,
                        relation_transformation=Transformation.LEAKY_RELU if regression else Transformation.SIGMOID)
    dataset = FileDataset(f'{data_path}/examples.txt', f'{data_path}/queries.txt')
    built_samples = template.build(settings).build_dataset(dataset)
    return built_samples


def train_model(built_samples, template, regression=True):
    logging.log(logging.INFO, "training model")
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
    logging.log(logging.INFO, "plotting predictions")
    data = confusion_matrix(target_labels, predicted_labels)
    figure(figsize=(20, 20))
    ax = seaborn.heatmap(data, annot=True, square=True, cmap='Blues', annot_kws={"size": 7}, cbar_kws={"shrink": 0.5})
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.savefig('confusion.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # import neuralogic
    # neuralogic.initialize(debug_mode=True)

    # domain = "blocksworld"
    domain = "satellite"
    numeric = False

    json_data = load_json(domain, numeric=numeric)
    problems, predicates, actions = parse_domain(json_data)
    data_path = export_problems(problems, domain)

    logging.log(logging.INFO, "building template")
    # template = satellite_regression_template(predicates, dim=3)
    template = basic_regression_template(predicates, dim=3)
    # template.draw("./imgs/template.png")

    built_samples = build_model(data_path, template)
    # built_samples[0].draw("./imgs/sample.png")

    target_labels, predicted_labels = train_model(built_samples, template)
    plot_predictions(target_labels, predicted_labels)
