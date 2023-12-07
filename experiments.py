import warnings
from dataclasses import dataclass
from os import listdir
from timeit import default_timer as timer

from torch_geometric.nn import GCNConv, RGCNConv, GATv2Conv, RGATConv

from encoding import Object2ObjectMultiGraph, Object2AtomBipartiteMultiGraph, Atom2AtomMultiGraph, \
    Object2AtomMultiGraph, Atom2AtomHigherOrderGraph, ObjectPair2ObjectPairMultiGraph, Object2ObjectGraph, \
    Object2AtomGraph, Object2AtomBipartiteGraph, Atom2AtomGraph, ObjectPair2ObjectPairGraph
from hashing import DistanceHashing
from modelsTorch import get_compatible_model, GINEConvWrap, MyException, GINConvWrap
from parsing import get_datasets


@dataclass
class Result:
    domain: str
    instance: str
    encoding: str
    model: str
    num_layers: int

    all_collisions: int
    bad_collisions: int
    sample_compression: float
    class_compression: float


class Logger:
    def __init__(self, file, separator=","):
        self.file = open(file, "w", buffering=1)
        self.file.write(
            "domain, instance, samples, encoding, model, num_layers, "
            "all_pairwise_collisions, bad_pairwise_collisions, sample_compression, class_compression, "
            "instance_encoding_time, model_creation_time, predictions_eval_time, my exception, other error")
        self.file.write("\n")

    def log_setting(self, domain, instance, samples, encoding, model, num_layers):
        self.file.write(
            ", ".join([domain, instance.name, str(samples), encoding.__name__, model.__name__, str(num_layers)]) + ", ")

    def log_results(self, all_collisions, bad_collisions, sample_compression, class_compression,
                    instance_encoding_time, model_encoding_time, collision_eval_time):
        self.file.write(", ".join(
            [str(all_collisions), str(bad_collisions), f"{sample_compression:.5f}", f"{class_compression:.5f}",
             f"{instance_encoding_time:.5f}", f"{model_encoding_time:.5f}", f"{collision_eval_time:.5f}"]))
        self.file.write("\n")

    def log_err(self, err):
        if isinstance(err, MyException):
            self.file.write("," * 7)
        else:
            self.file.write("," * 8)
        self.file.write(str(err))
        self.file.write("\n")

    def close(self):
        self.file.close()


def run_folder(folder, encodings, gnns, layer_nums, log_file, hidden_dim=8):
    logger = Logger(log_file)
    for domain in listdir(folder):
        print("==============" + domain + "==============")
        instances = get_datasets(folder + domain, descending=False)
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
                    print(f"{err=}, {type(err)=}")
                    continue
                all_models_eval_time = timer()
                prev_gnn = None
                for gnn_type in gnns:
                    for num_layers in layer_nums:
                        logger.log_setting(domain, instance, len(samples), encoding, gnn_type, num_layers)
                        try:
                            model_encoding_timer = timer()
                            model = get_compatible_model(samples, model_class=gnn_type, num_layers=num_layers,
                                                         hidden_channels=hidden_dim, previous_model=prev_gnn)
                            model_encoding_time = timer() - model_encoding_timer
                            # print(f'model_encoding_time: {model_encoding_time}')

                            prev_gnn = gnn_type

                            collision_eval_timer = timer()
                            distance_hashing = DistanceHashing(model, samples)

                            all_pairwise_collisions, _ = distance_hashing.get_all_collisions()
                            bad_pairwise_collisions, _ = distance_hashing.get_bad_collisions()
                            sample_compression, class_compression = distance_hashing.get_compression_rates()

                            collision_eval_time = timer() - collision_eval_timer
                            # print(f'collision_eval_time: {collision_eval_time}')

                            logger.log_results(all_pairwise_collisions, bad_pairwise_collisions,
                                               sample_compression, class_compression,
                                               instance_encoding_time, model_encoding_time, collision_eval_time)

                        except MyException as err:
                            logger.log_err(err)
                            # print(str(err))
                        except Exception as err:
                            logger.log_err(err)
                            raise err
                            warnings.warn(str(err))
                            print(f"{err=}, {type(err)=}")
                print(f"all_models_eval_time: {timer() - all_models_eval_time}")
    logger.close()


# %%

# encodings = [Object2ObjectMultiGraph, Object2AtomMultiGraph, Object2AtomBipartiteMultiGraph,
#              Atom2AtomMultiGraph, ObjectPair2ObjectPairMultiGraph, Atom2AtomHigherOrderGraph]
encodings = [Object2ObjectGraph, Object2ObjectMultiGraph, Object2AtomBipartiteMultiGraph, Object2AtomGraph, Object2AtomBipartiteGraph,
             Atom2AtomGraph, Atom2AtomMultiGraph, Atom2AtomHigherOrderGraph]
# encodings = [Atom2AtomGraph, ObjectPair2ObjectPairGraph]
# convs = [GINConvWrap, GCNConv, GINEConvWrap, GATv2Conv, RGCNConv]
convs = [GINEConvWrap, RGCNConv]

# layers = [2, 8]
layers = [4]

run_folder('./datasets/all/', encodings, convs, layers, "./results/results_all_rest4.csv")
