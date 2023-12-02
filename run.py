import pprint

from torch_geometric.nn import GCNConv, SAGEConv, RGCNConv, GATv2Conv, HGTConv

from data_structures import Object2ObjectGraph, Object2ObjectMultiGraph, Object2AtomGraph, Object2AtomBipartiteGraph, \
    Object2ObjectHeteroGraph
from hashing import DistanceHashing
from modelsTorch import SimpleGNN, BipartiteGNN, GINConvWrap, HeteroGNN
from parsing import get_datasets

# %% choose a dataset source
folder = "./datasets/rosta/blocks"
# folder = "./datasets/rosta/rovers"
# folder = "./datasets/rosta/transport"

datasets = get_datasets(folder, limit=1, descending=False)  # smallest dataset
# datasets = get_datasets(folder, limit=1, descending=True)  # largest dataset

dataset = datasets[0]

# %% add info about types, static facts, goal...

dataset.enrich_states(add_types=True, add_facts=True, add_goal=True)

# %%  1) choose an encoding

samples = dataset.get_samples(Object2ObjectGraph)
# samples = dataset.get_samples(Object2ObjectMultiGraph)
# samples = dataset.get_samples(Object2AtomGraph)
# samples = dataset.get_samples(Object2AtomBipartiteGraph)
# samples = dataset.get_samples(Object2ObjectHeteroGraph)

layout = samples[0].draw(symbolic=True)
samples[0].draw(symbolic=False, pos=layout)

# %% 2) choose a model

model = SimpleGNN(samples[0], model_class=GCNConv, num_layers=3)
# model = SimpleGNN(samples[0], model_class=SAGEConv, num_layers=3)
# model = SimpleGNN(samples[0], model_class=GINConvWrap, num_layers=3)
# model = SimpleGNN(samples[0], model_class=GATv2Conv, num_layers=3)
# model = SimpleGNN(samples[0], model_class=RGCNConv, num_layers=3)


# model = BipartiteGNN(samples[0], model_class=SAGEConv, num_layers=3)
# model = BipartiteGNN(samples[0], model_class=GATv2Conv, num_layers=3)

# model = HeteroGNN(samples[0], model_class=SAGEConv, num_layers=3) # not working yet
# model = HeteroGNN(samples[0], model_class=HGTConv, num_layers=3)

# model = GNN(samples)    # LRNN

# %% ...and test the expressiveness of the setup

distance_hashing = DistanceHashing(model, samples)

collisions = distance_hashing.get_all_collisions()
# collisions = distance_hashing.get_bad_collisions()

print("====Indistinguishable states detected:=====")
pprint.pprint(collisions)
print("Resulting [class] and [sample] compression rates:")
print(distance_hashing.get_compression_rates())
