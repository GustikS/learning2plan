from neuralogic.core import R, Settings, Template, Transformation, V, Aggregation
from neuralogic.nn.module import GATv2Conv, GCNConv, GINConv, SAGEConv, GENConv, SGConv, TAGConv

from policy_rules.util.template_settings import neuralogic_settings


def basic_template(predicates, dim=10, num_layers=3, actions=None, classification=True):
    if classification:
        assert actions is not None

    template = Template()

    if actions:
        template += action_rules(actions, predicates, dim, num_layers)

    template += anonymous_predicates(predicates, dim)

    template += object_info_aggregation(max(predicates.values()), dim)
    # template += atom_info_aggregation(max(predicates.values()), dim)

    template += object2object_edges(max(predicates.values()), dim, "edge")

    # template += custom_message_passing("edge", "h0", dim)
    template += gnn_message_passing("edge", dim, num_layers=num_layers)
    # template += gnn_message_passing(f"{2}-ary", dim, num_layers=num_layers)

    # template += objects2atoms_exhaustive_messages(predicates, dim, num_layers=num_layers)
    # template += objects2atoms_anonymized_messages(max(predicates.values()), dim, num_layers=num_layers)

    if classification:
        pass
    else:
        template += final_pooling(dim, layers=range(1, num_layers + 2))

    return template


def action_rules(actions, predicates, dim, num_layers, ILG=True, merge_ilg=True, action_messages=False):
    """Of course this is not the only way to incorporate actions in the learning templates..."""
    # todo integrate also action effects - they could inform the template better
    if ILG:
        if merge_ilg:
            rules = []
            for predicate in predicates:
                if predicate.startswith("ag_") or predicate.startswith("ap_"):
                    raw_pred = predicate[3:]
                    variables = [f"X{ar}" for ar in range(predicates[predicate])]
                    head = R.get(raw_pred)(variables)
                    # infer the original predicate with a numeric transformation based on the prefix
                    body = [R.get(predicate)(variables)[predicate[0:2]:dim, ]]
                    # and extend it with the latent representation of each involved object
                    body += [R.get(f'h_{num_layers}')(var)[dim, dim] for var in variables]
                    rules.append(head <= body)
            rules.extend([action.to_rule(dim=dim) for action in actions])
        else:
            rules = [action.to_rule(predicate_prefix=pref, dim=dim) for action in actions for pref in ['ag_', 'ap_']]
    else:
        rules = [action.to_rule(dim=dim) for action in actions]

    if action_messages:
        for action in actions:
            predicates[action.name] = len(action.parameters)  # just extend the predicates with the action heads...

    return rules


def anonymous_predicates(predicates, dim, input_dim=1):
    """
    map all the domain predicates to newly derived ones (anonymous) while respecting the same arity
    *input_dim* = 3 for our numeric encoding of the goal info into the predicates, or just 1 otherwise
    """
    rules = []
    # anonymizing/embedding all domain predicates
    for (predicate, arity) in predicates.items():
        variables = [f"X{ar}" for ar in range(arity)]
        rules.append(
            R.get(f"{arity}-ary_{0}")(variables)[dim, dim]
            <= R.get(f"{predicate}")(variables)[dim, input_dim]
        )
    return rules


def final_pooling(hidden, layers, query_name="distance"):
    """aggregate all relevant info from the computation graph for a final output"""
    rules = []
    for layer in layers:  # e.g. just aggregate object embeddings from the layers...
        rules.append(
            R.get(query_name)[1, hidden] <= R.get(f"h_{layer}")(V.X)[hidden, hidden]
        )
    return rules


def object_info_aggregation(max_arity, dim, layer=0, unary_only=False, add_nullary=True):
    """objects aggregate info from all the atoms they are associated with"""
    rules = []
    # only absorb unary predicates (typical "features")
    max_arity = (1 if unary_only else max_arity)
    for arity in range(0, max_arity + 1):
        variables = [f"X{ar}" for ar in range(arity)]
        # all objects calculate their embeddings by aggregating info from all associated atoms
        positions = (
            range(arity) if arity else [0] if add_nullary else []
        )  # optionally add also nullary atoms here
        rules += [
            R.get(f"h_{layer + 1}")(f"X{i}")[dim, dim]
            <= R.get(f"{arity}-ary_{layer}")(variables)[dim, dim]
            for i in positions
        ]
    return rules


def atom_info_aggregation(max_arity, dim, layer=0):
    """vice-versa, atoms aggregate info from all the objects they contain"""
    rules = []
    for arity in range(0, max_arity + 1):
        variables = [f"X{ar}" for ar in range(arity)]
        rules.append(
            R.get(f"{arity}-ary_{layer + 1}")(variables)[dim, dim]
            <= [R.get(f"h_{layer + 1}")(f"X{i}")[dim, dim] for i in range(arity)]
            + [R.get(f"{arity}-ary_{layer}")(variables)[dim, dim]]
        )
    return rules


def object2object_edges(max_arity, dim, edge_name="edge"):
    """i.e. constructing the GAIFMAN graph's binary relation (derived/anonymous)"""
    rules = []
    for arity in range(0, max_arity + 1):
        variables = [f"X{ar}" for ar in range(arity)]
        pairs = (
            (i, j) for i in variables for j in variables if i != j
        )  # all pairwise interactions
        rules += [
            R.get(edge_name)(pair)[dim, dim]
            <= R.get(f"{arity}-ary_{0}")(variables)[dim, dim]
            for pair in pairs
        ]
    return rules


def objects2atoms_anonymized_messages(max_arity, dim, num_layers=3):
    """i.e. something like message-passing on the bipartite object-atom (ILG,munin,...) graph representation,
    while using the derived (anonymous) relations"""
    rules = []
    for layer in range(num_layers):
        rules += object_info_aggregation(max_arity, dim, layer)
    for layer in range(num_layers - 1):
        rules += atom_info_aggregation(max_arity, dim, layer)
    return rules


def objects2atoms_exhaustive_messages(predicates, dim, num_layers=3, object_name="h"):
    """i.e. even closer to something like GNNs on the bipartite (ILG,munin,...) graph representation,
    passing messages on the ORIGINAL relations (as opposed to the anonymized ones, which is more compact)
    """
    rules = []
    # anonymizing/embedding all domain predicates
    for (predicate, arity) in predicates.items():
        if not arity:
            continue  # here we just skip the nullary atoms
        variables = [f"X{i}" for i in range(arity)]
        for layer in range(1, num_layers):
            # objects -> atom
            rules.append(
                R.get(f"h_{predicate}_{layer}")(variables)[dim, dim]
                <= [
                    R.get(f"{object_name}_{layer}")(f"X{i}")[dim, dim]
                    for i in range(arity)
                ]
                + [R.get(f"_{predicate}")(variables)]
            )
            # atom => objects
            rules += [
                R.get(f"{object_name}_{layer}")(f"X{i}")[dim, dim]
                <= R.get(f"h_{predicate}_{layer}")(variables)[dim, dim]
                for i in range(arity)
            ]
    return rules


def atom2atom_messages(max_arity, dim, num_layers=3):
    # TODO the last remaining (classic) message-passing mode...
    pass


def custom_message_passing(binary_relation, unary_relation, dim, layer=1, bidirectional=True):
    """just a custom rule for passing a message/features (unary_relation) along a given binary relation (binary_relation)"""
    rules = []
    rules.append(
        R.get(f"h{layer}")(V.X)[dim, dim]
        <= R.get(binary_relation)(V.X, V.Y)[dim, dim]
        & R.get(unary_relation)(V.Y)[dim, dim]
    )
    if bidirectional:
        rules.append(
            R.get(f"h{layer}")(V.X)[dim, dim]
            <= R.get(binary_relation)(V.Y, V.X)[dim, dim]
            & R.get(unary_relation)(V.Y)[dim, dim]
        )
    return rules


def gnn_message_passing(binary_relation, dim, gnn_type="SAGE", num_layers=3):
    """classic message passing reusing some existing GNN models as implemented in LRNN rules..."""

    match gnn_type:
        case "SAGE":
            model_class = SAGEConv
        case "GIN":
            model_class = GINConv
        case "TAG":
            model_class = TAGConv
        case _:
            raise NotImplementedError

    rules = []
    for layer in range(1, num_layers + 1):
        rules += model_class(in_channels=dim,
                             out_channels=dim,
                             output_name=f"h_{layer + 1}",
                             feature_name=f"h_{layer}",
                             edge_name=binary_relation,
                             activation=neuralogic_settings.relation_transformation,
                             aggregation=neuralogic_settings.rule_aggregation)()
    return rules


def build_template(template, regression=True, compression=True, pruning=True):
    settings = Settings(
        iso_value_compression=compression,
        rule_transformation=(
            Transformation.LEAKY_RELU if regression else Transformation.TANH
        ),
        relation_transformation=(
            Transformation.LEAKY_RELU if regression else Transformation.SIGMOID
        ),
        chain_pruning=pruning
    )
    model = template.build(settings)
    return model
