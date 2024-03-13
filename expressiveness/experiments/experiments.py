import os.path
import sys
import warnings
from os import listdir
from timeit import default_timer as timer

from torch_geometric.nn import GCNConv, SAGEConv, RGCNConv, GATv2Conv, GENConv, FiLMConv, HGTConv, HANConv

from ..encoding import Object2ObjectMultiGraph, Object2AtomBipartiteMultiGraph, Atom2AtomMultiGraph, \
    Object2AtomMultiGraph, Atom2AtomHigherOrderGraph, Object2ObjectGraph, \
    Object2AtomGraph, Object2AtomBipartiteGraph, Atom2AtomGraph
from ..hashing import DistanceHashing
from ...learning.modelsTorch import get_compatible_model, GINEConvWrap, MyException, GINConvWrap
from ...parsing import get_datasets


class Logger:
    def __init__(self, file, separator=","):
        self.file = open(file, "w", buffering=1)
        self.file.write(
            "domain, instance, samples, encoding, model, num_layers, aggr, hidden_dim, "
            "all_pairwise_collisions, bad_pairwise_collisions, near_collisions, sample_compression, class_compression, "
            "instance_encoding_time, model_creation_time, predictions_eval_time, my exception, other error")
        self.file.write("\n")

    def log_setting(self, domain, instance, samples, encoding, model, num_layers, aggr, hidden_dim):
        self.file.write(
            ", ".join([domain, instance.name, str(samples), encoding.__name__, model.__name__,
                       str(num_layers), aggr, str(hidden_dim)]) + ", ")

    def log_results(self, all_collisions, bad_collisions, near_collisions, sample_compression, class_compression,
                    instance_encoding_time, model_encoding_time, collision_eval_time):
        self.file.write(", ".join(
            [str(all_collisions), str(bad_collisions), str(near_collisions),
             f"{sample_compression:.5f}", f"{class_compression:.5f}",
             f"{instance_encoding_time:.5f}", f"{model_encoding_time:.5f}", f"{collision_eval_time:.5f}"]))
        self.file.write("\n")

    def log_err(self, err):
        if isinstance(err, MyException):
            self.file.write("," * 9)
        else:
            self.file.write("," * 10)
        self.file.write(str(err))
        self.file.write("\n")

    def close(self):
        self.file.close()


def run_domain(domain_folder, encodings, gnns, logger, layer_nums=[4], aggrs=["add"], hidden_dims=[8]):
    print("==============" + domain_folder + "==============")
    instances = get_datasets(domain_folder, descending=False)
    for instance in instances:
        print("==========" + instance.name)
        instance.enrich_states(add_types=True, add_facts=True, add_goal=True)
        for encoding in encodings:
            try:
                instance_encoding_timer = timer()
                samples = instance.get_samples(encoding)
                instance_encoding_time = timer() - instance_encoding_timer
                print(f'{encoding.__name__} - encoding_time: {instance_encoding_time}', end=" ")
            except Exception as err:
                warnings.warn(str(err))
                # raise err
                print(f"{err=}, {type(err)=}")
                continue
            all_models_eval_time = timer()
            prev_gnn = None
            for gnn_type in gnns:
                for num_layers in layer_nums:
                    for aggr in aggrs:
                        for hidden_dim in hidden_dims:
                            logger.log_setting(domain_folder.split("/")[-1], instance, len(samples), encoding,
                                               gnn_type, num_layers, aggr, hidden_dim)
                            try:
                                model_encoding_timer = timer()
                                model = get_compatible_model(samples, model_class=gnn_type, num_layers=num_layers,
                                                             hidden_channels=hidden_dim, aggr=aggr,
                                                             previous_model=prev_gnn)
                                model_encoding_time = timer() - model_encoding_timer
                                # print(f'model_encoding_time: {model_encoding_time}')

                                prev_gnn = gnn_type

                                collision_eval_timer = timer()
                                distance_hashing = DistanceHashing(model, samples, epsilon_check=False)

                                all_pairwise_collisions, _ = distance_hashing.get_all_collisions()
                                bad_pairwise_collisions, _ = distance_hashing.get_bad_collisions()
                                sample_compression, class_compression = distance_hashing.get_compression_rates()
                                near_collisions = distance_hashing.epsilon_sanity_check()

                                collision_eval_time = timer() - collision_eval_timer
                                # print(f'collision_eval_time: {collision_eval_time}')

                                logger.log_results(all_pairwise_collisions, bad_pairwise_collisions, near_collisions,
                                                   sample_compression, class_compression,
                                                   instance_encoding_time, model_encoding_time, collision_eval_time)

                            except MyException as err:
                                logger.log_err(err)
                                # raise err
                                # print(str(err))
                            except Exception as err:
                                logger.log_err(err)
                                # raise err
                                warnings.warn(str(err))
                                print(f"{err=}, {type(err)=}")
            print(f"all_models_eval_time: {timer() - all_models_eval_time}")


# %%

encodings = [Object2ObjectGraph, Object2ObjectMultiGraph,
             Object2AtomGraph, Object2AtomMultiGraph, Object2AtomBipartiteGraph, Object2AtomBipartiteMultiGraph,
             Atom2AtomGraph, Atom2AtomMultiGraph,
             # Object2ObjectHeteroGraph, Object2AtomHeteroGraph, Atom2AtomHeteroGraph,
             Atom2AtomHigherOrderGraph]
# encodings = [Atom2AtomHigherOrderGraph, ObjectPair2ObjectPairGraph, ObjectPair2ObjectPairMultiGraph]  # long runtime
# encodings = [ObjectPair2ObjectPairMultiGraph]

convs = [GCNConv, SAGEConv, GINConvWrap, GATv2Conv, GINEConvWrap, GENConv, RGCNConv, FiLMConv, HGTConv, HANConv]    # all
# convs = [SAGEConv, GENConv, RGCNConv, FiLMConv, NNConvWrap, GeneralConv]  # supports aggr
# convs = [RGCNConv, FiLMConv, HGTConv, HANConv]  # hetero
# convs = [NNConvWrap, TransformerConv, PDNConv, GeneralConv]  # new ones

layers = [4]
# layers = [2, 8, 16]  # 16 is too much for adding

aggregations = ["add", "mean", "max"]
# aggregations = ["add", "mean"]
# aggregations = ["add"]

# hidden = [8, 32, 128]
hidden = [8]


def main(source_folder, experiment_name, **kwargs):
    target_file = "./results/" + experiment_name + "_" + source_folder.split("/")[-1] + ".csv"
    logger = Logger(target_file)
    source_items = sorted(listdir(source_folder))
    if os.path.isdir(source_folder + "/" + source_items[0]):
        for domain in source_items:
            run_domain(source_folder + "/" + domain, encodings, convs, logger, layer_nums=layers, aggrs=aggregations,
                       hidden_dims=hidden)
    else:
        run_domain(source_folder, encodings, convs, logger, layer_nums=layers, aggrs=aggregations,
                   hidden_dims=hidden)
    logger.close()


if __name__ == "__main__":

    args = sys.argv[1:]
    source_folder = args[0]
    try:
        experiment_name = args[1]
    except:
        experiment_name = "test2"
    # convs = [SAGEConv]
    main(source_folder, experiment_name, kwargs=args)
