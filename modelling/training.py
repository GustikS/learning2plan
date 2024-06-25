import argparse
import logging
import pickle
import time

import seaborn
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import neuralogic

if not neuralogic.is_initialized():
    # neuralogic.initialize()
    # # neuralogic.initialize(jar_path="../jar/NeuraLogic-maven.jar", debug_mode=False)
    neuralogic.initialize(jar_path="../jar/NeuraLogic.jar", debug_mode=False)  # custom momentary backend upgrades

from neuralogic.core import Settings
from neuralogic.dataset import FileDataset
from neuralogic.nn import get_evaluator
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.optim import Adam

from samples import export_problems, parse_domain, get_filename
from sklearn.metrics import confusion_matrix
from templates import basic_template, build_template

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(filename)s:%(lineno)s] %(message)s",
)


def build_samples(model, data_path):
    logging.log(logging.INFO, "building model samples")
    dataset = FileDataset(f"{data_path}/examples.txt", f"{data_path}/queries.txt")
    dataset = model.ground(dataset)
    built_samples = model.build_dataset(dataset)
    return built_samples, dataset


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


def train(domain, numeric, save_file=None, plotting=False, problem_limit=-1, classification=True):
    built_samples, model, template, _ = prepare_experiment(domain, numeric, problem_limit=problem_limit,
                                                           classification=classification)

    target_labels, predicted_labels = predict(built_samples, template, train=True)

    store_weights(model, save_file)

    store_template(template, save_file)

    if plotting:
        plot_predictions(target_labels, predicted_labels)

    return model, template


def store_weights(model, save_file):
    if save_file is not None:
        state_dict = model.state_dict()
        # state_dict["weight_names"] = {
        #     k: str(v) for k, v in state_dict["weight_names"].items()
        # }
        # with open(save_file, 'wb') as f:
        #     pickle.dump(state_dict, f)
        torch.save(state_dict, f'{save_file}_weights')
        logging.log(logging.INFO, f"Model weights saved to {save_file}_weights")


def store_template(template, save_file):
    file = open(save_file + "_template", 'wb')
    pickle.dump(template, file)
    file.close()


def prepare_experiment(domain, numeric, export_lrnn_files=True, draw=True, problem_limit=-1, classification=True):
    problems, predicates, actions = parse_domain(domain, numeric, problem_limit=problem_limit)
    model, template = prepare_model(predicates, actions, draw, classification=classification)
    if export_lrnn_files:
        data_path = export_problems(problems, domain, numeric)
    else:
        data_path = get_filename(domain, numeric, "lrnn", "../", "")
    built_samples, ground_samples = build_samples(model, data_path)
    if draw:
        built_samples[0].draw("./imgs/sample.png")
    return built_samples, model, template, ground_samples


def prepare_model(predicates, actions=None, draw=True, classification=True):
    """This is where a model gets assembled for the current workflow"""
    # template = satellite_regression_template(predicates, dim=3)
    template = basic_template(predicates, dim=3, actions=actions, classification=classification)

    model = build_template(template, compression=not draw)
    if draw:
        template.draw("./imgs/template.png")
    return model, template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", type=str, default="blocksworld", choices=["satellite", "blocksworld"])
    parser.add_argument("--numeric", type=bool, default=False)  # keep numeric false for now...
    parser.add_argument("--save_file", type=str, default='./target/stored_model')
    args = parser.parse_args()
    domain_name = args.domain
    numeric = args.numeric
    save_file = args.save_file + f'_{domain_name}'
    print(f"{domain_name=}")
    print(f"{numeric=}")
    print(f"{save_file=}")

    model, template = train(domain_name, numeric, save_file=save_file, plotting=True, classification=True)
