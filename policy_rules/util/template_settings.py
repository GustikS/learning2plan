import os
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Union

import neuralogic
from neuralogic.core import Aggregation, Settings, Template, Transformation
from neuralogic.nn.java import NeuraLogic
from termcolor import colored

# we can set up all the learning/numeric-evaluation-related settings here
neuralogic_settings = Settings(
    iso_value_compression=True,
    chain_pruning=True,
    rule_transformation=Transformation.TANH,  # change to RELU for better training
    rule_aggregation=Aggregation.AVG,  # change to avg for better generalization
    relation_transformation=Transformation.TANH,  # change to RELU for better training - check label match
    epochs=100,
)

# neuralogic_settings["inferOutputFcns"] = False
neuralogic_settings["oneQueryPerExample"] = False
neuralogic_settings["preprocessTemplateInference"] = False


"""
DZC 15/07/2024. See commit b24e3b918277778f05585f6c378cb411b72c13e8 for original code
Main changes: 
- pack template and weights into one file, but keep text representation of template into another file
- change typing of load_stored_model to NeuraLogic from Union[NeuraLogic, Template] as the old code always seemed to return NeuraLogic or None
"""


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
