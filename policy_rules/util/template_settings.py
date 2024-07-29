import os
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Union

from neuralogic.core import Aggregation, Settings, Template, Transformation
from neuralogic.nn.init import Constant, Uniform
from neuralogic.nn.java import NeuraLogic
from termcolor import colored

global neuralogic_settings


def debug_settings():
    global neuralogic_settings
    neuralogic_settings = Settings(
        # for baseline policy
        initializer=Constant(value=1),
        rule_transformation=Transformation.IDENTITY,
        rule_aggregation=Aggregation.MIN,
        relation_transformation=Transformation.IDENTITY,
        iso_value_compression=False,
        chain_pruning=True,
    )


def train_settings():
    global neuralogic_settings
    neuralogic_settings = Settings(
        initializer=Uniform(),
        rule_transformation=Transformation.TANH,
        rule_aggregation=Aggregation.AVG,
        relation_transformation=Transformation.TANH,
        iso_value_compression=True,  # set to true for training speedup (but bad for debugging the NN images)
        chain_pruning=True,
        # rule_transformation=Transformation.LEAKY_RELU,
        # relation_transformation=Transformation.LEAKY_RELU,
    )


# the default is settings for actual learning
train_settings()

# neuralogic_settings["inferOutputFcns"] = False
neuralogic_settings["oneQueryPerExample"] = False
neuralogic_settings["preprocessTemplateInference"] = False
# this was only good for the lifted queries setting (not used in the current setup)
neuralogic_settings["aggregateConflictingQueries"] = False


@dataclass
class SaveData:
    model_state_dict: Optional[Dict]
    template: Template


def load_stored_model(template_path: str) -> NeuraLogic:
    if not os.path.exists(template_path):
        print(f"WARNING: Model {template_path} does not exist")
        return None
    with open(template_path, "rb") as f:
        save_data = pickle.load(f)
        model_state_dict = save_data.model_state_dict
        template = save_data.template
    model = template.build(neuralogic_settings)
    if model_state_dict:
        model.load_state_dict(model_state_dict)
    return model


def save_template_model(model: Union[NeuraLogic | Template], save_model_path: str) -> None:
    if "/" in save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    model_state_dict = None
    if isinstance(model, NeuraLogic):
        model_state_dict = model.state_dict()
        template = Template()
        template.template = model.source_template
    else:
        template = model

    with open(save_model_path, "wb") as f:
        save_data = SaveData(model_state_dict, template)
        pickle.dump(save_data, f)

    with open(save_model_path + "_template.txt", "w") as f:
        f.write(str(template))

    print(colored(f"Model saved to {save_model_path}", "green"))


def load_model_weights(model: NeuraLogic, weights_file: str = None):
    try:
        f = open(weights_file, "rb")
        weights = pickle.load(f)
        model.load_state_dict(weights)
        print(f"Loaded also stored weights from {weights_file}")
    except (IOError, OSError) as e:
        print(f"No stored weights found at {weights_file}): {e}")
    except Exception as e:
        print(f"Problem loading weights from {weights_file}: {e}")
        f.close()
