import warnings
from dataclasses import dataclass
from os import listdir

from torch_geometric.nn import GCNConv, RGCNConv

from encoding import Object2ObjectMultiGraph, Object2AtomBipartiteMultiGraph, Atom2AtomMultiGraph
from hashing import DistanceHashing
from modelsTorch import get_compatible_model, GINEConvWrap, MyException
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
        self.file = open(file, "w")
        self.file.write(
            "domain, instance, encoding, model, num_layers, all_collisions, bad_collisions, sample_compression, class_compression, my exception, other error")
        self.file.write("\n")

    def log_setting(self, domain, instance, encoding, model, num_layers):
        self.file.write(", ".join([domain, instance.name, encoding.__name__, model.__name__, str(num_layers)]) + ", ")

    def log_results(self, all_collisions, bad_collisions, sample_compression, class_compression):
        self.file.write(
            ", ".join([str(all_collisions), str(bad_collisions), str(sample_compression), str(class_compression)]))
        self.file.write("\n")

    def log_err(self, err):
        if isinstance(err, MyException):
            self.file.write("," * 4)
        else:
            self.file.write("," * 5)
        self.file.write(str(err))
        self.file.write("\n")

    def close(self):
        self.file.close()


def run_folder(folder, encodings, gnns, layer_nums, log_file, hidden_dim=8):
    logger = Logger(log_file)
    for domain in listdir(folder):
        instances = get_datasets(folder + domain, descending=False)
        for instance in instances:
            instance.enrich_states(add_types=True, add_facts=True, add_goal=True)
            for encoding in encodings:
                samples = instance.get_samples(encoding)
                prev_gnn = None
                for gnn_type in gnns:
                    for num_layers in layer_nums:
                        logger.log_setting(domain, instance, encoding, gnn_type, num_layers)
                        try:
                            model = get_compatible_model(samples, model_class=gnn_type, num_layers=num_layers,
                                                         hidden_channels=hidden_dim, previous_model=prev_gnn)
                            prev_gnn = gnn_type

                            distance_hashing = DistanceHashing(model, samples)

                            all_collisions = len(distance_hashing.get_all_collisions())
                            bad_collisions = len(distance_hashing.get_bad_collisions())
                            sample_compression, class_compression = distance_hashing.get_compression_rates()

                            logger.log_results(all_collisions, bad_collisions, sample_compression, class_compression)

                        except MyException as err:
                            logger.log_err(err)
                        except Exception as err:
                            logger.log_err(err)
                            warnings.warn(str(err))
                            print(f"{err=}, {type(err)=}")
    logger.close()


# %%

encodings = [Object2ObjectMultiGraph, Object2AtomBipartiteMultiGraph, Atom2AtomMultiGraph]
convs = [GCNConv, GINEConvWrap, RGCNConv]
layers = [2, 8]

run_folder('./datasets/rosta/', encodings, convs, layers, "./results/results.csv")
