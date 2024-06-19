import argparse
import logging
import time

import seaborn
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import neuralogic

if not neuralogic.is_initialized():
    neuralogic.initialize()

from neuralogic.core import Settings
from neuralogic.dataset import FileDataset
from neuralogic.nn import get_evaluator
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.optim import Adam

from samples import export_problems, parse_domain, get_filename
from sklearn.metrics import confusion_matrix
from templates import basic_regression_template, get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(filename)s:%(lineno)s] %(message)s",
)


def build_samples(model, data_path):
    logging.log(logging.INFO, "building model samples")
    dataset = FileDataset(f"{data_path}/examples.txt", f"{data_path}/queries.txt")
    built_samples = model.build_dataset(dataset)
    return built_samples


def predict(built_samples, template, train: bool, regression=True):
    logging.log(logging.INFO, "training model")
    settings = Settings(
        optimizer=Adam(lr=0.001),
        epochs=100,
        error_function=MSE() if regression else CrossEntropy()
    )
    evaluator = get_evaluator(template, settings)

    if train:
        for i, (current_total_loss, number_of_samples) in enumerate(evaluator.train(built_samples.samples)):
            logging.log(logging.INFO,
                        f"epoch: {i} total loss: {current_total_loss} samples updated: {number_of_samples}")

    target_labels, predicted_labels = [], []
    for sample, prediction in zip(built_samples.samples, evaluator.test(built_samples)):
        logging.log(logging.INFO, f"Target: {sample.target}, Predicted: {round(prediction)} ({prediction})")
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


def train(domain, numeric, save_file=None, plotting=False):
    built_samples, model, template = prepare_experiment(domain, numeric)

    target_labels, predicted_labels = predict(built_samples, template, train=True)

    if save_file is not None:
        state_dict = model.state_dict()
        state_dict["weight_names"] = {
            k: str(v) for k, v in state_dict["weight_names"].items()
        }
        torch.save(state_dict, save_file)
        logging.log(logging.INFO, f"Model saved to {save_file}")

    if plotting:
        plot_predictions(target_labels, predicted_labels)


def load(domain, numeric, save_file, plotting=False):
    built_samples, model, template = prepare_experiment(domain, numeric)

    model.load_state_dict(torch.load(save_file))
    target_labels, predicted_labels = predict(built_samples, template, train=False)

    if plotting:
        plot_predictions(target_labels, predicted_labels)


def prepare_experiment(domain, numeric, export_lrnn_files=True, draw=True):
    problems, predicates, actions = parse_domain(domain, numeric)
    model, template = prepare_model(predicates, actions)
    if export_lrnn_files:
        data_path = export_problems(problems, domain, numeric)
    else:
        data_path = get_filename(domain, numeric, "lrnn", "../", "")
    built_samples = build_samples(model, data_path)
    if draw:
        built_samples[1].draw("./imgs/sample.png")
    return built_samples, model, template


def prepare_model(predicates, actions=None, draw=True):
    """This is where a model gets assembled for the current workflow"""
    # template = satellite_regression_template(predicates, dim=3)
    template = basic_regression_template(predicates, dim=3, actions=actions)
    model = get_model(template)
    if draw:
        template.draw("./imgs/template.png")
    return model, template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", type=str, default="blocksworld", choices=["satellite", "blocksworld"])
    parser.add_argument("--numeric", type=bool, default=False)
    parser.add_argument("--save_file", type=str, default=None)
    args = parser.parse_args()
    domain_name = args.domain
    numeric = args.numeric
    save_file = args.save_file
    print(f"{domain_name=}")
    print(f"{numeric=}")
    print(f"{save_file=}")

    train(domain_name, numeric, save_file=save_file, plotting=True)
    # load(domain_name, numeric, save_file=save_file)
