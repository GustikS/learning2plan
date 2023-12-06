import pprint

from torch_geometric.nn import GCNConv, SAGEConv, RGCNConv, GATv2Conv, HGTConv, HANConv, FiLMConv, RGATConv, GINEConv

from encoding import Object2ObjectGraph, Object2ObjectMultiGraph, Object2AtomGraph, Object2AtomBipartiteGraph, \
    Object2ObjectHeteroGraph, Object2AtomMultiGraph, Object2AtomBipartiteMultiGraph, Object2AtomHeteroGraph, \
    Atom2AtomGraph, Atom2AtomMultiGraph, Atom2AtomHeteroGraph, ObjectPair2ObjectPairGraph, \
    ObjectPair2ObjectPairMultiGraph, Atom2AtomHigherOrderGraph

from hashing import DistanceHashing
from modelsTorch import PlainGNN, BipartiteGNN, GINConvWrap, HeteroGNN, get_compatible_model, GINEConvWrap
from parsing import get_datasets

# %% choose a dataset source

# folder = "./datasets/rosta/debug"

folder = "./datasets/rosta/blocks"
# folder = "./datasets/rosta/rovers"
# folder = "./datasets/rosta/transport"

datasets = get_datasets(folder, limit=1, descending=False)  # smallest dataset
# datasets = get_datasets(folder, limit=1, descending=True)  # largest dataset

dataset = datasets[0]

# %% add info about types, static facts, goal...

dataset.enrich_states(add_types=True, add_facts=False, add_goal=True)

# %%  1) choose an encoding

# encoding = Object2ObjectGraph
# encoding = Object2ObjectMultiGraph
# encoding = Object2ObjectHeteroGraph
# encoding = Object2AtomGraph
# encoding = Object2AtomMultiGraph
# encoding = Object2AtomBipartiteGraph
# encoding = Object2AtomBipartiteMultiGraph
# encoding = Object2AtomHeteroGraph
# encoding = Atom2AtomGraph
# encoding = Atom2AtomMultiGraph
# encoding = Atom2AtomHeteroGraph

# encoding = ObjectPair2ObjectPairGraph
# encoding = ObjectPair2ObjectPairMultiGraph

encoding = Atom2AtomHigherOrderGraph

samples = dataset.get_samples(encoding)

# %% optional sample drawing for debugging purposes

layout = samples[0].draw(symbolic=True)
samples[0].draw(symbolic=False, pos=layout)

# %% 2) choose a model

# gnn_type = SAGEConv   # no edge attribute support
# gnn_type = GINConvWrap    # no edge attribute support

# gnn_type = GCNConv    # scalar edge weights supported
gnn_type = GATv2Conv  # edge attributes only in (normalized) attention coefficients
# gnn_type = GINEConvWrap   # edge attributes summed up with node attributes

# gnn_type = RGCNConv   # separate edge types (multi-relational) parameterization support
# gnn_type = FiLMConv  # separate edge types (multi-relational) parameterization support
# gnn_type = RGATConv  # separate edge types (multi-relational) parameterization support

# gnn_type = HGTConv    # general hetero-graph support, but reduces to bipartite multi-relational in our case
# gnn_type = HANConv    # general hetero-graph support, but reduces to bipartite multi-relational in our case

model = get_compatible_model(samples, model_class=gnn_type, num_layers=3, hidden_channels=8)

# %% ...and test the expressiveness of the setup

distance_hashing = DistanceHashing(model, samples)

collisions = distance_hashing.get_all_collisions()
# collisions = distance_hashing.get_bad_collisions()

print("====Indistinguishable states detected:=====")
pprint.pprint(collisions)
print("Resulting [class] and [sample] compression rates:")
print(distance_hashing.get_compression_rates())
