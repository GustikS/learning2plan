from neuralogic.core import Settings, R
from neuralogic.dataset import FileDataset
from neuralogic.nn import get_evaluator

from templates import basic_regression_template


def generic_relational_feature():
    return R.distance <= R.edge('X', 'Y') & R.edge('Y', 'Z') & R.edge('X', 'Z')


def adhoc_relational_feature():
    return R.distance <= R.on_board('Ins', 'Sat') & R.pointing('Sat', 'Star') & R.calibration_target('Ins', 'Star')


# %%
def dillons_rules():
    return [
        R.instrument_config('I', 'M', 'S', 'D') <=
        R.supports('I', 'M') & R.on_board('I', 'S') & R.calibration_target('I', 'D'),

        R.turn_to('S', 'D', 'D_new') <=
        R.ug_have_image('D', 'M') & R.instrument_config('I', 'M', 'S', 'D'),

        R.switch_on('I', 'S') <=
        R.ug_have_image('D', 'M') & R.pointing('S', 'D') & R.instrument_config('I', 'M', 'S', 'D') & ~R.power_on('I'),

        R.calibrate('S', 'I', 'D') <=
        R.ug_have_image('D', 'M') & R.pointing('S', 'D') & R.instrument_config('I', 'M', 'S', 'D') & ~R.calibrated('I'),

        R.take_image('S', 'D', 'I', 'M') <=
        R.ug_have_image('D', 'M') & R.instrument_config('I', 'M', 'S', 'D'),

        R.turn_to('S', 'D', 'D_new') <=
        R.ug_pointing('S', 'D') & ~R.ug_have_image('D_other', 'M') & R.instrument_config('I', 'M', 'S', 'D')

    ]


def eval_examples(dataset, template, experiment=""):
    settings = Settings(iso_value_compression=False)
    model = template.build(settings)
    built_samples = model.build_dataset(dataset)

    evaluator = get_evaluator(template, settings)
    outputs = evaluator.test(built_samples, generator=False)
    print("experiment: " + experiment)
    print(outputs)

    for i, sample in enumerate(built_samples):
        sample.draw(f"counter_sample{i}_{experiment}.png")


# %%

dataset = FileDataset(examples_file="../datasets/lrnn/satellite/counter_examples.txt",
                      queries_file="../datasets/lrnn/satellite/counter_queries.txt")

predicates = {"on_board": 2, "supports": 2, "calibration_target": 2, "power_avail": 1, "pointing": 2}

# %% these counterexamples are indeed indistinguishable with classic template(s)

template = basic_regression_template(predicates, dim=1, num_layers=1)
template.draw("template_basic.png")
eval_examples(dataset, template, "basic")

# %% let's increase the (GNN-like) template's expressiveness a bit by adding certain graphlet(s)

template += generic_relational_feature()
template.draw("template_graphlet.png")
eval_examples(dataset, template, "graphlets")

# %% add the relational feature - of course this is cheating, just showing (but such relational features can come e.g. from Treeliker)

template = basic_regression_template(predicates, dim=1, num_layers=1)
template += adhoc_relational_feature()
template.draw("template_custom.png")
eval_examples(dataset, template, "custom")


# %%