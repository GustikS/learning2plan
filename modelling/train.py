import argparse
import logging

import seaborn
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from neuralogic.core import R, Settings, Template, Transformation, V
from neuralogic.dataset import FileDataset
from neuralogic.nn import get_evaluator
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.optim import Adam
from samples import export_problems, load_json, parse_domain
from sklearn.metrics import confusion_matrix
from templates import basic_regression_template, get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(filename)s:%(lineno)s] %(message)s",
)


def build_model(model, data_path):
    logging.log(logging.INFO, "building model")
    dataset = FileDataset(f"{data_path}/examples.txt", f"{data_path}/queries.txt")
    built_samples = model.build_dataset(dataset)
    return built_samples


def predict(built_samples, template, train: bool, regression=True):
    logging.log(logging.INFO, "training model")
    settings = Settings(
        optimizer=Adam(lr=0.001),
        epochs=100,
        error_function=MSE() if regression else CrossEntropy(),
    )
    evaluator = get_evaluator(template, settings)

    if train:
        for i, (current_total_loss, number_of_samples) in enumerate(
            evaluator.train(built_samples.samples)
        ):
            print(
                f"epoch: {i} total loss: {current_total_loss} samples updated: {number_of_samples}"
            )

    target_labels, predicted_labels = [], []
    for sample, prediction in zip(built_samples.samples, evaluator.test(built_samples)):
        print(f"Target: {sample.target}, Predicted: {round(prediction)} ({prediction})")
        target_labels.append(sample.target), predicted_labels.append(round(prediction))

    return target_labels, predicted_labels


def plot_predictions(target_labels, predicted_labels):
    logging.log(logging.INFO, "plotting predictions")
    data = confusion_matrix(target_labels, predicted_labels)
    figure(figsize=(20, 20))
    ax = seaborn.heatmap(
        data,
        annot=True,
        square=True,
        cmap="Blues",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.5},
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.savefig("confusion.png", bbox_inches="tight")
    plt.show()


def parse_data(domain, numeric):
    json_data = load_json(domain, numeric=numeric)
    problems, predicates, actions = parse_domain(json_data)
    return problems, predicates, actions


def train(domain, numeric, save_file=None):
    data = parse_data(domain, numeric)
    problems, predicates, actions = data

    # template = satellite_regression_template(predicates, dim=3)
    template = basic_regression_template(predicates, dim=3)
    # template.draw("./imgs/template.png")

    model = get_model(template)

    data_path = export_problems(problems, domain)
    built_samples = build_model(model, data_path)
    # built_samples[0].draw("./imgs/sample.png")

    target_labels, predicted_labels = predict(built_samples, template, train=True)

    if save_file is not None:
        state_dict = model.state_dict()
        state_dict["weight_names"] = {
            k: str(v) for k, v in state_dict["weight_names"].items()
        }
        torch.save(state_dict, save_file)
        logging.log(logging.INFO, f"Model saved to {save_file}")

    # plot_predictions(target_labels, predicted_labels)


def load(domain, numeric, save_file):
    data = parse_data(domain, numeric)
    problems, predicates, actions = data

    # template = satellite_regression_template(predicates, dim=3)
    template = basic_regression_template(predicates, dim=3)
    # template.draw("./imgs/template.png")

    model = get_model(template)
    model.load_state_dict(torch.load(save_file))

    data_path = export_problems(problems, domain)
    built_samples = build_model(model, data_path)
    # built_samples[0].draw("./imgs/sample.png")

    target_labels, predicted_labels = predict(built_samples, template, train=False)

    plot_predictions(target_labels, predicted_labels)


if __name__ == "__main__":
    # import neuralogic
    # neuralogic.initialize(debug_mode=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", type=str, default="satellite")
    parser.add_argument("--numeric", action="store_true")
    parser.add_argument("--save_file", type=str, default=None)
    args = parser.parse_args()
    domain = args.domain
    numeric = args.numeric
    save_file = args.save_file
    print(f"{domain=}")
    print(f"{numeric=}")
    print(f"{save_file=}")

    # train(domain, numeric, save_file=save_file)
    load(domain, numeric, save_file=save_file)
