import os
import pickle
from typing import Union

from neuralogic.core import Template, Settings, Transformation, Aggregation
from neuralogic.nn.java import NeuraLogic

# we can setup all the learning/numeric-evaluation-related settings here
neuralogic_settings = Settings(
    iso_value_compression=True,
    chain_pruning=True,
    rule_transformation=Transformation.TANH,  # change to RELU for better training
    rule_aggregation=Aggregation.SUM,  # change to avg for better generalization
    relation_transformation=Transformation.SIGMOID,  # change to RELU for better training - check label match
    epochs=100
)


def load_stored_model(template_path: str) -> Union[NeuraLogic | Template]:
    _stored_template = None
    try:
        f = open(template_path + "_template", 'rb')
        _stored_template = pickle.load(f)
        assert isinstance(_stored_template, Template)
        print(f"Loaded a stored template {template_path}_template")
    except (IOError, OSError) as e:
        print(f"No stored template found at {template_path}_template): {e}")
        return _stored_template
    except Exception as e:
        print(f"Problem loading a template from {template_path}_template: {e}")
        f.close()
    if _stored_template:
        model = _stored_template.build(neuralogic_settings)
        load_model_weights(model, template_path + "_weights")
        return model


def store_template_model(model: Union[NeuraLogic | Template], save_model_path: str):
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    if isinstance(model, NeuraLogic):
        with open(save_model_path + "_weights", 'wb') as f:
            pickle.dump(model.state_dict(), f)
        template = Template()
        template.template = model.source_template
    else:
        template = model

    with open(save_model_path + "_template", 'wb') as f:
        pickle.dump(template, f)
    with open(save_model_path + "_template.txt", 'w') as f:
        f.write(str(template))


def load_model_weights(model: NeuraLogic, weights_file: str = None):
    try:
        f = open(weights_file, 'rb')
        weights = pickle.load(f)
        model.load_state_dict(weights)
        print(f"Loaded also stored weights from {weights_file}")
    except (IOError, OSError) as e:
        print(f"No stored weights found at {weights_file}): {e}")
    except Exception as e:
        print(f"Problem loading weights from {weights_file}: {e}")
        f.close()
