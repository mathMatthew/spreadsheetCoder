from typing import Any, Dict, Tuple, List
import networkx as nx
from collections import deque
import logging, datetime, json

import validation, errs
import conv_tracker as ct
import convert_xml as cxml
import conversion_rules as cr
import dag_tables as g_tables
from coding_centralized import convert_to_python_type

array1_functions = [
    "AND",
    "AVEDEV",
    "AVERAGE",
    "COUNT",
    "COUNTA",
    "COUNTBLANK",
    "MAX",
    "MEDIAN",
    "MIN",
    "MODE",
    "PERCENTILE",
    "PERCENTRANK",
    "PRODUCT",
    "SUM",
    "NPV",
]


def subset_graph(original_graph, outputs_to_keep: List[int]) -> nx.MultiDiGraph:
    new_graph = nx.MultiDiGraph()
    visited = set()
    new_inputs = set()

    def select_nodes_and_edges(node):
        if node in visited:
            return
        visited.add(node)
        # Add the node and its attributes to the new graph
        new_graph.add_node(node, **original_graph.nodes[node])
        if original_graph.nodes[node]["node_type"] == "input":
            new_inputs.add(node)
        for predecessor in original_graph.predecessors(node):
            # Iterate over each edge between predecessor and node
            for key in original_graph[predecessor][node]:
                # Add each edge and its attributes to the new graph
                new_graph.add_edge(
                    predecessor, node, key=key, **original_graph[predecessor][node][key]
                )
            select_nodes_and_edges(predecessor)

    # Select nodes and edges starting from the first two outputs
    for output in outputs_to_keep:
        select_nodes_and_edges(output)
    new_graph.graph["name"] = original_graph.graph["name"]
    sorted_inputs = sorted(
        new_inputs, key=lambda item: original_graph.graph["input_node_ids"].index(item)
    )
    sorted_outputs = sorted(
        outputs_to_keep,
        key=lambda item: original_graph.graph["output_node_ids"].index(item),
    )
    new_graph.graph["input_node_ids"] = sorted_inputs
    new_graph.graph["output_node_ids"] = sorted_outputs
    new_graph.graph["max_node_id"] = max(new_graph.nodes())

    assert validation.is_valid_graph(
        new_graph, False
    ), f"Graph {new_graph.graph['name']} is not valid. function: subset graph"

    return new_graph


def build_nx_graph(lxml_tree) -> nx.MultiDiGraph:
    max_node_id: int = 0
    tree_name = lxml_tree.getroot().attrib.get("name")

    # Create a directed graph
    G: nx.MultiDiGraph = nx.MultiDiGraph()

    # Prepare to store output node attributes and IDs
    output_attributes: Dict[int, Dict] = {}
    output_node_ids = set()
    input_node_ids = set()

    for output in lxml_tree.find("Outputs"):
        output_node_id: int = int(output.attrib["node_id"])
        del output.attrib["node_id"]
        output_node_ids.add(output_node_id)
        output_attributes[output_node_id] = dict(output.attrib)

    # Add nodes to the graph
    for node in lxml_tree.find("Nodes"):
        node_attrs = dict(node.attrib)
        node_id: int = int(node_attrs["node_id"])
        del node_attrs["node_id"]
        max_node_id = max(max_node_id, int(node_id))
        if node_attrs["node_type"] == "input":
            input_node_ids.add(node_id)
        if "input_order" in node_attrs:
            node_attrs["input_order"] = int(node_attrs["input_order"])

        # Merge output attributes, checking for overlaps
        if node_id in output_attributes:
            for key, value in output_attributes[node_id].items():
                if key in node_attrs and key != "data_type":
                    raise ValueError(
                        f"Attribute overlap detected for node {node_id}: '{key}' in tree: {tree_name}"
                    )
                elif key == "type":
                    if key in node_attrs:
                        if node_attrs[key] != value:
                            raise ValueError(
                                f"Type mismatch detected for node {node_id}: node type '{node_attrs[key]}' vs output type '{value}' in tree: {tree_name}"
                            )
                node_attrs[key] = value

        G.add_node(node_id, **node_attrs)

    # Add edges based on dependencies
    for dependency in lxml_tree.find("NodeDependencies"):
        parent_node = int(dependency.attrib["parent_node_id"])
        child_node = int(dependency.attrib["child_node_id"])
        parent_position = int(dependency.attrib["parent_position"])
        G.add_edge(parent_node, child_node, parent_position=parent_position)

    # Define the priority of node name types
    name_type_priority = {
        "alias": 1,
        "address": 2,
        "array_formula_parent_address": 3,
    }

    # Validate and add names for named nodes, considering the new priority
    for named_node in lxml_tree.find("NamedNodes"):
        node_attribs = G.nodes[int(named_node.attrib["node_id"])]
        current_name_type = node_attribs.get("node_name_type", None)
        new_name_type = named_node.attrib["node_name_type"]

        # Check if the new name type is recognized
        if new_name_type not in name_type_priority:
            raise ValueError(f"Unrecognized node_name_type: {new_name_type}")

        # Check if the current node name type has a lower priority (higher number) than the new name type
        if current_name_type is None or name_type_priority[
            new_name_type
        ] < name_type_priority.get(current_name_type, float("inf")):
            # Update with new name if new type has higher priority or if node doesn't have a name type yet
            node_attribs["node_name_type"] = new_name_type
            node_attribs["node_name"] = named_node.attrib["node_name"]
        elif new_name_type == current_name_type:
            # If it already has a name of the same type, raise an error.
            cxml.save_xml_and_raise(
                lxml_tree,
                f"Duplicate node_name_type detected. Node_id: {named_node.attrib['node_id']}, node_name_type: {current_name_type}",
            )

    # Set max_nodeG_id and name as attributes of the graph
    G.graph["max_node_id"] = max_node_id
    G.graph["name"] = tree_name
    sorted_inputs = sorted(
        input_node_ids, key=lambda node_id: G.nodes[node_id]["input_order"]
    )
    G.graph["input_node_ids"] = list(sorted_inputs)
    sorted_outputs = sorted(
        output_node_ids, key=lambda node_id: G.nodes[node_id]["output_order"]
    )
    G.graph["output_node_ids"] = list(sorted_outputs)

    return G


def node_location_description(G, node_id) -> str:
    node_id_desc = f"(node_id: {node_id})"
    attribs = G.nodes[node_id]
    if "function_name" in attribs:
        function_desc = f", the function: {attribs['function_name']}"
    else:
        function_desc = ""

    if "node_name" in attribs:
        desciption = f"The {attribs['node_name']} node {node_id_desc}{function_desc}"
        return desciption
    else:
        children = list(G.successors(node_id))
        if not children:
            desciption = f"The output node {attribs['output_name']} {node_id_desc}{function_desc}"
            return desciption
        else:
            # for the description let's just show one way to get to the node. so choice of child here is arbitrary
            child_id = children[0]
            edge_data = G.get_edge_data(node_id, child_id)
            first_key = list(edge_data.keys())[0]
            parent_position = edge_data[first_key]["parent_position"]
            desciption = f"{node_location_description(G, child_id)}, the {_ordinal(parent_position)} input, {node_id_desc}{function_desc}"
            return desciption


def _ordinal(number):
    # Convert 0-based to 1-based
    number += 1

    # Determine the suffix
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    # Check for 11-13 because they follow different rules
    if 10 <= number % 100 <= 13:
        suffix = "th"
    else:
        # Use the appropriate suffix; default to 'th'
        suffix = suffixes.get(number % 10, "th")

    return f"{number}{suffix}"


def topo_sort_subgraph(graph, nodes):
    """
    Perform a topological sort on a subgraph consisting of a specified subset of nodes.

    :param graph: The original directed acyclic graph (DAG).
    :param nodes: A list of nodes to include in the subgraph.
    :return: A list of nodes in topologically sorted order.
    """
    if not isinstance(graph, nx.DiGraph):
        raise ValueError("The graph must be a directed acyclic graph (DAG).")

    # Create a subgraph with the specified nodes
    subgraph = graph.subgraph(nodes)

    # Check if the subgraph is a DAG
    if not nx.is_directed_acyclic_graph(subgraph):
        raise ValueError("The subgraph is not a directed acyclic graph (DAG).")

    # Perform a topological sort on the subgraph
    return list(nx.topological_sort(subgraph))


def transform_from_to(
    base_dag,
    transform_from_to_dag,
    node_id_to_replace,
    match_mapping,
    tables_dict,
    conversion_rules,
    signature_definition_library,
    auto_add_signatures,
    conversion_tracker,
    separate_tables,
):

    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."
    assert validation.is_valid_base_graph(
        base_dag, True
    ), f"base_dag is not valid before transform {transform_from_to_dag.graph['name']}"

    # record action in conversion_tracker
    ct.update_conversion_tracker_record_transform(
        conversion_tracker, transform_from_to_dag.graph["name"]
    )
    # step1 : excise nodes and rewire node_id_to_replace.
    # 1.A: figure out which nodes to delete and which to rewire (which edges to delete and which to add)
    from_to_outputs = transform_from_to_dag.graph["output_node_ids"]
    from_dag: nx.MultiDiGraph = subset_graph(
        transform_from_to_dag, [from_to_outputs[0]]
    )
    to_dag: nx.MultiDiGraph = subset_graph(transform_from_to_dag, [from_to_outputs[1]])
    from_inputs = from_dag.graph["input_node_ids"]
    from_output = from_dag.graph["output_node_ids"][0]

    assert (
        node_id_to_replace == match_mapping[f"t-{from_output}"]
    ), "node_id_to_replace must be the mapped node from the output of the from_dag"

    edges_to_delete = list(base_dag.in_edges(node_id_to_replace, keys=True))

    edges_to_add = [
        (
            match_mapping[f"t-{input_node_id}"],
            node_id_to_replace,
            {"parent_position": parent_position},
        )
        for parent_position, input_node_id in enumerate(from_inputs)
    ]

    matching_b_nodes = [
        match_mapping[f"t-{node_id}"] for node_id in from_dag.nodes()
    ]  # same as if we just got values from dictionary for all keys beginning with t.

    potential_nodes_to_delete = [
        match_mapping[f"t-{node_id}"]
        for node_id in from_dag.nodes()
        if node_id != from_output and node_id not in from_inputs
    ]

    def can_delete_node(node, graph, allowed_successors):
        for successor in graph.successors(node):
            if successor not in allowed_successors:
                return False
        return True

    nodes_to_delete = []
    for del_candidate_id in potential_nodes_to_delete:
        if can_delete_node(del_candidate_id, base_dag, matching_b_nodes):
            nodes_to_delete.append(del_candidate_id)

    # Step 1.B: excise nodes and rewire
    # Apply deletions of edges
    for u, v, key in edges_to_delete:
        base_dag.remove_edge(u, v, key)

    base_dag.remove_nodes_from(nodes_to_delete)

    # Apply additions of new edges
    for u, v, attr in edges_to_add:
        base_dag.add_edge(u, v, **attr)

    # step2 : expand node and return new id of expanded node.
    new_id, skip_stack = expand_node(
        node_id_to_expand=node_id_to_replace,
        function_logic_dag=to_dag,
        base_dag=base_dag,
        tables_dict=tables_dict,
        signature_definitions=conversion_rules,
        signature_definition_library=signature_definition_library,
        auto_add_signatures=auto_add_signatures,
        conversion_tracker=conversion_tracker,
        separate_tables=separate_tables,
    )
    assert validation.is_valid_base_graph(
        base_dag, True
    ), f"base_dag is not valid after transform {transform_from_to_dag.graph['name']}"
    return new_id, skip_stack


def renumber_nodes(graph):
    def update_io_lists(mapping):
        for prop in ["input_node_ids", "output_node_ids"]:
            if prop in graph.graph:
                graph.graph[prop] = [
                    mapping[node_id]
                    for node_id in graph.graph[prop]
                    if node_id in mapping
                ]

    assert validation.is_valid_graph(
        graph, False
    ), "Graph is not valid prior to renumbering nodes"

    offset = graph.graph["max_node_id"] + 1
    temp_mapping = {node_id: node_id + offset for node_id in graph.nodes()}
    nx.relabel_nodes(graph, temp_mapping, copy=False)
    update_io_lists(temp_mapping)

    # Mapping from old IDs to new IDs
    final_mapping = {
        old_id: new_id for new_id, old_id in enumerate(graph.nodes(), start=1)
    }
    nx.relabel_nodes(graph, final_mapping, copy=False)
    update_io_lists(final_mapping)

    graph.graph["max_node_id"] = len(final_mapping)

    assert validation.is_valid_graph(
        graph, False
    ), "Graph is not valid after renumbering nodes"
    return graph


def remove_arrays(G, node_id, seen):
    if node_id in seen:
        return

    seen.add(node_id)

    # Collect all incoming edges and sort by parent_position
    incoming_edges = sorted(
        list(G.in_edges(node_id, data="parent_position", keys=True)), key=lambda x: x[3]
    )  # x[3] is the parent_position

    # List of edges to remove later
    edges_to_remove = []

    for parent_id, _, key, parent_position in incoming_edges:
        if G.nodes[parent_id].get("function_name", "").upper() == "ARRAY":
            # Collect and sort the grandparents by their parent_position
            grandparents = sorted(
                list(G.in_edges(parent_id, data="parent_position", keys=True)),
                key=lambda x: x[3],
            )

            # Ignore the first two grandparents (array dimension nodes)
            for grandparent_id, _, g_key, _ in grandparents[2:]:
                # Rewire: Connect the grandparent directly to the target node
                # Assign the current parent_position
                G.add_edge(
                    grandparent_id, node_id, key=g_key, parent_position=parent_position
                )

                # Increment the parent_position counter after each rewire
                parent_position += 1

            # Add current edge to removal list
            edges_to_remove.append((parent_id, node_id, key))

    # Remove marked edges and collect nodes for removal check
    nodes_to_check = set()
    for parent_id, child_id, key in edges_to_remove:
        G.remove_edge(parent_id, child_id, key)
        nodes_to_check.add(parent_id)

    # Check if the nodes have no outgoing edges and remove them if so
    # if no one else is using the node, delete it (which will also remove the edges between it an the original nodes grandchildren)
    for check_node_id in nodes_to_check:
        if G.out_degree(check_node_id) == 0:
            dimension_nodes = [
                gp[0]
                for gp in sorted(
                    list(G.in_edges(check_node_id, data="parent_position", keys=True)),
                    key=lambda x: x[3],
                )[:2]
            ]
            G.remove_node(check_node_id)
            for dim_node in dimension_nodes:
                if G.out_degree(dim_node) == 0:
                    G.remove_node(dim_node)


def convert_to_binomial(
    G,
    node_id,
    comm_func_dict,
    conversion_rules,
    signature_definition_library,
    auto_add_signatures,
    conversion_tracker,
):
    """
    Convert the given node to a graph of binomial functions, ensuring that all input data types are the same.
    Parameters:
    - G: the graph
    - node_id: the ID of the node to be converted
    - comm_func_dict: the dictionary entry for the commutative function
    - conversion_tracker: the conversion tracker
    """
    parent_data_types = cr.get_parent_data_types(G, node_id)
    assert (
        len(set(parent_data_types)) == 1
    ), f"Convert to binomail requires that all the inputs have the same data type. Data types are {format(set(parent_data_types))}"
    binary_function_name = comm_func_dict["bin_func"]
    new_dag = generate_binomial_dag(
        binary_function_name, len(parent_data_types), parent_data_types[0]
    )
    ct.update_conversion_tracker_record_binomial_expansion(
        conversion_tracker, G.nodes[node_id]["function_name"], comm_func_dict
    )
    new_id, skip_stack = expand_node(
        node_id_to_expand=node_id,
        function_logic_dag=new_dag,
        base_dag=G,
        tables_dict={},  # tables_dict is not needed given we just created it--and we didn't add tables
        signature_definitions=conversion_rules,
        signature_definition_library=signature_definition_library,
        auto_add_signatures=auto_add_signatures,
        conversion_tracker=conversion_tracker,
        separate_tables=False,  # see tables_dict comment above
    )
    return new_id, skip_stack


def update_dag_with_data_types(
    G: nx.MultiDiGraph,
    topo_sorted_nodes,
    signature_definitions: Dict,
    signature_definition_library: Dict,
    auto_add_signatures: bool,
    conversion_tracker: Dict,
):
    assert nx.is_directed_acyclic_graph(G), f'{G.graph["name"]} is not a DAG'
    assert validation.is_valid_conversion_rules_dict(
        signature_definitions
    ), "signature is not valid"
    assert validation.is_valid_signature_definition_dict(
        signature_definition_library, True, True
    ), "signature is not valid"

    for node_id in topo_sorted_nodes:
        if "data_type" in G.nodes[node_id]:
            continue
        if "function_name" not in G.nodes[node_id]:
            errs.save_dag_and_raise__node(
                G,
                node_id,
                f"Node id: {node_id} does not have data_type but is not a function.",
            )
            return

        data_types = cr.get_data_types(
            G,
            node_id,
            signature_definitions,
            signature_definition_library,
            auto_add_signatures,
            conversion_tracker,
        )
        if len(data_types) == 1:
            G.nodes[node_id]["data_type"] = data_types[0]
        elif len(data_types) > 1:
            G.nodes[node_id]["data_types"] = data_types
        else:
            raise Exception(f"get_data_types returned an empty list for Node {node_id}")


def identify_sub_graph_nodes(G, sub_graph_output_node_id):
    """
    Identify all nodes within the sub_graph defined by tracing upstream from the sub_graph_output_node_id.
    Utilizes a stack for iterative deepening to avoid recursion.
    """
    stack = [sub_graph_output_node_id]
    sub_graph_nodes = set()

    while stack:
        current_node = stack.pop()
        if current_node in sub_graph_nodes or G.nodes[current_node].get(
            "persist", False
        ):
            continue
        sub_graph_nodes.add(current_node)
        stack.extend(G.predecessors(current_node))

    return sub_graph_nodes


def calculate_upstream_counts_toposort(G, sub_graph_nodes):
    """
    Calculate upstream counts for nodes in the sub_graph using a topological sort approach,
    focusing the calculation on the sub_graph defined by the sub_graph_nodes parameter.
    """
    sub_graph = G.subgraph(sub_graph_nodes)

    # Initialize function nodes with 1, others 0, ensuring we only include nodes within the sub_graph
    step_counts = {
        node: 1 if sub_graph.nodes[node]["node_type"] == "function" else 0
        for node in sub_graph.nodes()
    }

    # Perform topological sort on the subgraph and calculate counts
    for node in nx.topological_sort(sub_graph):
        step_counts[node] += sum(
            step_counts.get(pred, 0) for pred in sub_graph.predecessors(node)
        )

    return step_counts


def persist_sub_graph_where_optimal(
    G, output_node_id, step_count_trade_off, prohibited_types
):
    sub_graph_nodes = identify_sub_graph_nodes(G, output_node_id)
    step_counts = calculate_upstream_counts_toposort(G, sub_graph_nodes)

    # Filter nodes to exclude those with prohibited data types
    permissible_nodes = [
        node
        for node in sub_graph_nodes
        if G.nodes[node]["data_type"] not in prohibited_types
    ]

    # Calculate potential step count savings, considering only permissible nodes
    step_count_saves = {
        node: step_counts[node] * (len(list(G.successors(node))) - 1)
        for node in permissible_nodes
    }

    if not step_count_saves:
        return 0  # No permissible nodes to consider for persisting

    # Cache the node with the minimum savings > step_count_trade_off
    # the reason I want to persist the node with the Least eligible savings (vs the max) is due to the
    # iterative nature of this. Caching the one with the max (as I had been doing) has the
    # potential to in a later round become trivial as it could be that in later round a prior
    # node gets persisted and this first one ends up doing very little. By persisting instead one
    # one just over the minimum savings, we avoid that problem.

    # Filter nodes that have savings greater than step_count_trade_off
    eligible_nodes = {
        node: saves
        for node, saves in step_count_saves.items()
        if saves > step_count_trade_off
    }

    if eligible_nodes:
        min_savings_node = min(eligible_nodes, key=eligible_nodes.get)
        # Cache the node with the minimum savings
        G.nodes[min_savings_node]["persist"] = True
        return min_savings_node
    else:
        return 0


def persist_where_optimal(G, step_count_trade_off, prohibited_types):
    nodes_to_check = []
    for node_id in G.nodes:
        if G.nodes[node_id].get("persist", False) or "output_name" in G.nodes[node_id]:
            nodes_to_check.append(node_id)

    stack = deque(nodes_to_check)
    while stack:
        node_id = stack.popleft()
        return_val = persist_sub_graph_where_optimal(
            G, node_id, step_count_trade_off, prohibited_types
        )
        if return_val:
            # stack.append(return_val) #no longer needed with new strategy.
            stack.append(node_id)


def reduce_sub_graph_to_threshold(
    G, sub_graph_output_node_id, max_step_count, prohibited_types
):
    # 1. Identify nodes in the subgraph and calculate step counts
    sub_graph_nodes = identify_sub_graph_nodes(G, sub_graph_output_node_id)
    step_counts = calculate_upstream_counts_toposort(G, sub_graph_nodes)

    # Check if the total step count is already within the threshold
    if sum(step_counts.values()) <= max_step_count:
        return  # The total step count is within threshold; no action needed

    # 2. Calculate the target step count
    integer_component, modulo_component = divmod(
        step_counts[sub_graph_output_node_id], max_step_count
    )
    target_step_count = step_counts[sub_graph_output_node_id] / (integer_component + 1)

    # 3. Filter out nodes with prohibited data types after determining the target
    permissible_nodes = [
        node
        for node in sub_graph_nodes
        if G.nodes[node]["data_type"] not in prohibited_types
    ]

    # Adjust step counts to only include permissible nodes
    permissible_step_counts = {node: step_counts[node] for node in permissible_nodes}

    # If there are no permissible nodes to persist, raise an error
    if not permissible_nodes:
        raise ValueError(
            "Unable to reduce step count within threshold: no permissible nodes to persist."
        )
        return

    # 4. Find the node closest to the target step count for persisting among permissible nodes
    node_to_persist = min(
        permissible_nodes,
        key=lambda node: (
            abs(permissible_step_counts[node] - target_step_count),
            -permissible_step_counts[
                node
            ],  # Break ties by preferring larger step counts
        ),
    )

    # Cache the selected node and call recursively for the remaining sub_graph (the sub_graph_output_node_id)
    G.nodes[node_to_persist]["persist"] = True
    reduce_sub_graph_to_threshold(
        G, sub_graph_output_node_id, max_step_count, prohibited_types
    )


def reduce_all_sub_graphs_to_threshold(G, max_step_count, prohibited_types):
    nodes_to_check_and_reduce = []
    for node_id in G.nodes:
        if G.nodes[node_id].get("persist", False):
            continue
        nodes_to_check_and_reduce.append(node_id)

    for node_id in nodes_to_check_and_reduce:
        reduce_sub_graph_to_threshold(G, node_id, max_step_count, prohibited_types)


def calculate_branch_depth_and_persist(
    G, max_branching_depth, conversion_rules, prohibited_types
):
    """
    Modifies the graph G by persisting nodes based on branching depth, considering prohibited data types.

    NOTE:
    Requires that the signatures in conversion_rules uses the key "branching_function"
    with value True, where applicable, to count branching.

    Parameters:
    - G (nx.MultiDiGraph): The directed graph.
    - max_branching_depth (int): The maximum allowed branching depth before persisting is forced.
    - conversion_rules (dict): Conversion rules, including signatures for identifying branching functions.
    - prohibited_types (list): Data types that are prohibited from being persisted.

    Raises:
    - ValueError: If a node exceeds the max_branching_depth and has a data type that is prohibited from persisting.
    """
    # Initialize branch depth dictionary
    branch_depths = {node_id: 0 for node_id in G.nodes()}

    # Traverse the graph in reverse topological order
    for node_id in reversed(list(nx.topological_sort(G))):
        node_attribs = G.nodes[node_id]

        # Nodes that aren't functions have branch_depth of 1 by definition
        if not "function_name" in node_attribs:
            branch_depths[node_id] = 1
            continue

        # Same goes for nodes that are persisted
        if node_attribs["persist"]:
            branch_depths[node_id] = 1
            continue

        # And output nodes
        if "output_order" in node_attribs:
            branch_depths[node_id] = 1
            continue

        function_signature = cr.match_first_signature__node(
            G, node_id, conversion_rules
        )

        current_depth = max(
            (branch_depths[successor] for successor in G.successors(node_id)), default=1
        )

        # Adjust branch depth calculation for branching nodes, based on matching signature
        current_depth += (
            1 if function_signature.get("branching_function", False) else 0
        )  # Increment depth for branching

        # Check if current depth exceeds max allowed
        if current_depth > max_branching_depth:
            # If the node has a prohibited data type, raise an error
            if node_attribs["data_type"] in prohibited_types:
                raise ValueError(
                    f"Add support for using branching functions with prohibited data types. Node ID: {node_id}"
                )
            # Otherwise, persist the node
            G.nodes[node_id]["persist"] = True
            branch_depths[node_id] = 1  # Reset depth after persisting
        else:
            branch_depths[node_id] = current_depth


def persist_node_or_predecessors(G, node_id, prohibited_types):
    if G.nodes[node_id]["data_type"] in prohibited_types:
        for pred_id in G.predecessors(node_id):
            persist_node_or_predecessors(G, pred_id, prohibited_types)
    else:
        G.nodes[node_id]["persist"] = True

    return


def mark_nodes_to_persist_by_usage_count(G, usage_count_threshold, prohibited_types):
    for node_id in G.nodes:
        if G.nodes[node_id].get("function_name"):
            if len(list(G.successors(node_id))) > usage_count_threshold:
                persist_node_or_predecessors(G, node_id, prohibited_types)


def mark_nodes_to_persist(
    G,
    conversion_rules,
    prohibited_types=[],
    all_outputs=False,
    all_array_nodes=False,
    step_count_trade_off=150,
    branching_threshold=0,
    total_steps_threshold=1000,
    usage_count_threshold=0,  # deprecated, use step_count_trade_off instead
):
    """
    Marks nodes for persisting ensuring not to persist nodes with prohibited data types.

    Parameters:
    - G (nx.MultiDiGraph): The directed graph.
    - all_outputs (bool): Whether to mark all outputs for persisting.
    - all_array_nodes (bool): Whether to mark all array nodes for persisting.
    - branching_threshold (int): Cache node if branching depth is greater than branching_threshold. Set to 0 to not use.
    - usage_count_threshold (int): Deprecated, use step_count_trade_off instead.
    - step_count_trade_off (int): Step count savings threshold for persisting. Set to 0 to not use. This one is preferred for optimization.
    - total_steps_threshold (int): Considering non-persisted nodes as step, persist node to prevent step-count > threshold for any persisted nodes. Set to 0 to not use.
    - conversion_rules (dict): Conversion rules dictionary.
    - prohibited_types (list): Data types prohibited from being persisted.
    """
    assert validation.is_valid_graph(
        G, True
    ), "Graph is not valid at start of mark nodes for persisting"

    # Helper function to check for prohibited types and raise an error
    def check_and_raise_for_prohibited_types(node_id, rule):
        if G.nodes[node_id]["data_type"] in prohibited_types:
            raise ValueError(
                f"Node {node_id} required to be persisted by rule {rule}, but has a prohibited data type for persisting."
            )

    # 1. Mark all output nodes for persisting if flagged
    if all_outputs:
        for node_id in G.graph["output_node_ids"]:
            check_and_raise_for_prohibited_types(
                node_id, "Cache Outputs"
            )  # Check for prohibited types
            if G.nodes[node_id]["node_type"] in ("input", "function"):
                G.nodes[node_id]["persist"] = True
        assert validation.is_valid_graph(
            G, True
        ), "Graph is not valid. Mark nodes for persisting. After 1"

    # 2. Mark all array nodes for persisting if flagged
    if all_array_nodes:
        for node_id in G.nodes:
            if G.nodes[node_id].get("function_name", "").upper() == "ARRAY":
                check_and_raise_for_prohibited_types(
                    node_id, "Cache Array Nodes"
                )  # Check for prohibited types
                G.nodes[node_id]["persist"] = True
        assert validation.is_valid_graph(
            G, True
        ), "Graph is not valid. Mark nodes for persisting. After 2"

    # 3. Function-specific persisting requirements
    for node_id in G.nodes:
        node_type = G.nodes[node_id].get("node_type")
        if node_type == "function":
            function_signature = cr.match_first_signature__node(
                G, node_id, conversion_rules
            )
            if function_signature:
                if function_signature.get("requires_persist", False) or (
                    "template" in function_signature
                    and conversion_rules["templates"][
                        function_signature["template"]
                    ].get("force-persist", False)
                ):
                    check_and_raise_for_prohibited_types(
                        node_id,
                        f'Signature {G.nodes[node_id]["function_name"]} Requires Caching',
                    )  # Check for prohibited types
                    G.nodes[node_id]["persist"] = True
    assert validation.is_valid_graph(
        G, True
    ), "Graph is not valid. Mark nodes for persisting. After 3"

    # 4. Step count trade-off persisting
    if step_count_trade_off > 0:
        persist_where_optimal(G, step_count_trade_off, prohibited_types)
        assert validation.is_valid_graph(
            G, True
        ), "Graph is not valid. Mark nodes for persisting. After 6"

    # 5. Usage count threshold persisting -- deprecated
    if usage_count_threshold > 0:
        mark_nodes_to_persist_by_usage_count(G, usage_count_threshold, prohibited_types)
        assert validation.is_valid_graph(
            G, True
        ), "Graph is not valid. Mark nodes for persisting. After 5"

    # 6. Branching depth threshold persisting
    if branching_threshold > 0:
        calculate_branch_depth_and_persist(
            G, branching_threshold, conversion_rules, prohibited_types
        )
        assert validation.is_valid_graph(
            G, True
        ), "Graph is not valid. Mark nodes for persisting. After 4"

    # 7. Total steps threshold persisting
    if total_steps_threshold > 0:
        reduce_all_sub_graphs_to_threshold(G, total_steps_threshold, prohibited_types)

    assert validation.is_valid_graph(
        G, True
    ), "Graph is not valid after mark nodes for persisting."


def generate_transforms_categories(
    transforms_dict: Dict[str, nx.MultiDiGraph]
) -> Tuple[Dict[str, List[nx.MultiDiGraph]], Dict[str, List[nx.MultiDiGraph]]]:
    transforms_from_to: Dict[str, List[nx.MultiDiGraph]] = {}
    transforms_protect: Dict[str, List[nx.MultiDiGraph]] = {}

    for name, transform_logic_dag in transforms_dict.items():
        outputs = transform_logic_dag.graph["output_node_ids"]
        original_output_count: int = len(outputs)

        # Handle protect transforms
        for i in range(2, original_output_count):
            output_node_id_to_keep = outputs[i]
            transform_protect: nx.MultiDiGraph = subset_graph(
                transform_logic_dag, [output_node_id_to_keep]
            )
            protect_function_name = transform_protect.nodes[output_node_id_to_keep][
                "function_name"
            ]
            if protect_function_name not in transforms_protect:
                transforms_protect[protect_function_name] = []
            transforms_protect[protect_function_name].append(transform_protect)

        # Handle from-to transforms
        if original_output_count >= 2:
            from_to = subset_graph(transform_logic_dag, outputs[:2])
            from_function_name = from_to.nodes[outputs[0]]["function_name"]
            if from_function_name not in transforms_from_to:
                transforms_from_to[from_function_name] = []
            transforms_from_to[from_function_name].append(from_to)

    return transforms_from_to, transforms_protect


def convert_graph(
    dag_to_convert: nx.MultiDiGraph,
    conversion_rules: Dict[str, Any],
    signature_definition_library: Dict[str, Any],
    auto_add_signatures: bool,
    conversion_tracker: Dict[str, Any],
    tables_dict: Dict[str, Dict[str, Any]],
    separate_tables: bool,
    renum_nodes: bool,
) -> None:
    """
    Converts a directed acyclic graph (DAG) (the dag_to_convert) into a DAG transformed
    by applying the defined transforms and the function_logic_dags.
    """
    assert validation.is_valid_base_graph(dag_to_convert, False), "dag is not valid"
    assert validation.is_valid_conversion_rules_dict(
        conversion_rules
    ), "conversion rules dictionary is not valid"
    assert validation.is_valid_signature_definition_dict(
        signature_definition_library, False, True
    ), "signature is not valid"

    if separate_tables:
        g_tables.separate_named_tables(
            dag_to_convert, dag_to_convert.nodes(), tables_dict
        )

    function_logic_dags: Dict[str, nx.MultiDiGraph] = conversion_rules[
        "function_logic_dags"
    ]
    transforms_from_to, transforms_protect = generate_transforms_categories(
        conversion_rules["transforms"]
    )

    func_logic_sigs = cr.create_signature_dictionary(function_logic_dags)
    cr.add_signatures_to_library(
        func_logic_sigs, signature_definition_library, "function_logic_dag", True
    )

    update_dag_with_data_types(
        G=dag_to_convert,
        topo_sorted_nodes=nx.topological_sort(dag_to_convert),
        signature_definitions=conversion_rules,
        signature_definition_library=signature_definition_library,
        auto_add_signatures=auto_add_signatures,
        conversion_tracker=conversion_tracker,
    )
    assert validation.is_valid_base_graph(dag_to_convert, True), "dag is not valid"

    def add_to_queue(node_ids):
        for node_id in node_ids:
            if node_id not in seen:
                stack.append(node_id)
                seen.add(node_id)

    protect_nodes_dict = {}

    def add_to_protect_nodes_dict(matching_nodes_mapping, protect_dag):
        protect_from_dag_name = protect_dag.graph["name"]
        for key, node_id in matching_nodes_mapping.items():
            # add all of the based dag IDs by finding the keys for transforms that map to them.
            if key.startswith("t-"):  #
                protect_nodes_dict[node_id].append(protect_from_dag_name)

    # Initialize the stack with output nodes
    stack = deque(dag_to_convert.graph["output_node_ids"])
    skip_stack: List[int] = []
    seen = set(stack)  # Set to keep track of nodes already in the stack

    array1_nodes_that_were_checked_for_arrays = set()
    function_logic_dags_with_types_added = set()

    while stack:
        assert validation.is_valid_base_graph(
            dag_to_convert, True
        ), "converted dag is not valid"
        node_id = stack.pop()
        if node_id in skip_stack:
            continue

        # Process the node
        if dag_to_convert.nodes[node_id]["node_type"] != "function":
            continue
        function_name = dag_to_convert.nodes[node_id]["function_name"]

        # if array1function, remove array nodes and rewire relevant grandchildren directly to node.
        if function_name in array1_functions:
            remove_arrays(
                dag_to_convert, node_id, array1_nodes_that_were_checked_for_arrays
            )

        # check if node function is in commutative_functions_to_convert_to_binomial
        # if so, convert to binomial
        commutative_functions_to_convert_to_binomial = conversion_rules.get(
            "commutative_functions_to_convert_to_binomial", {}
        )
        if function_name in commutative_functions_to_convert_to_binomial:
            comm_func_dict = commutative_functions_to_convert_to_binomial[function_name]
            new_ids, new_skip_stack = convert_to_binomial(
                G=dag_to_convert,
                node_id=node_id,
                comm_func_dict=comm_func_dict,
                conversion_rules=conversion_rules,
                signature_definition_library=signature_definition_library,
                auto_add_signatures=auto_add_signatures,
                conversion_tracker=conversion_tracker,
            )
            add_to_queue(new_ids)
            skip_stack.extend(new_skip_stack)
            continue

        # Check if the node is safe from a transform
        if function_name in transforms_protect:
            for protect_dag in transforms_protect[function_name]:
                protect_id: int = protect_dag.graph["output_node_ids"][
                    0
                ]  # by definition a protect has only one output node
                match_mapping = dict_of_matching_node_ids(
                    transform_dag=protect_dag,
                    base_dag=dag_to_convert,
                    transform_node_id=protect_id,
                    base_node_id=node_id,
                )
                if match_mapping:
                    # set protection
                    add_to_protect_nodes_dict(match_mapping, protect_dag)

        # Check if the node is a transform
        continue_to_while = False
        if function_name in transforms_from_to:
            for transform_logic_dag in transforms_from_to[function_name]:
                # check if this node is protected from this transform
                transform_name = transform_logic_dag.graph["name"]
                if transform_name not in protect_nodes_dict.get(node_id, []):
                    # if the current node matches the From tree in the transform logic dag
                    transform_from_node_id = transform_logic_dag.graph[
                        "output_node_ids"
                    ][0]
                    match_mapping: Dict[str, int] = dict_of_matching_node_ids(
                        transform_dag=transform_logic_dag,
                        base_dag=dag_to_convert,
                        transform_node_id=transform_from_node_id,
                        base_node_id=node_id,
                    )
                    if match_mapping:
                        new_ids, new_skip_stack = transform_from_to(
                            base_dag=dag_to_convert,
                            transform_from_to_dag=transform_logic_dag,
                            node_id_to_replace=node_id,
                            match_mapping=match_mapping,
                            tables_dict=tables_dict,
                            conversion_rules=conversion_rules,
                            signature_definition_library=signature_definition_library,
                            auto_add_signatures=auto_add_signatures,
                            conversion_tracker=conversion_tracker,
                            separate_tables=separate_tables,
                        )
                        add_to_queue(new_ids)
                        skip_stack.extend(new_skip_stack)
                        continue_to_while = True
                        break

        if continue_to_while:
            continue_to_while = False
            continue

        if function_name in function_logic_dags:
            function_logic_dag = function_logic_dags[function_name]
            if (
                function_logic_dag.graph["name"]
                not in function_logic_dags_with_types_added
            ):
                update_dag_with_data_types(
                    G=function_logic_dag,
                    topo_sorted_nodes=list(nx.topological_sort(function_logic_dag)),
                    signature_definitions=conversion_rules,
                    signature_definition_library=signature_definition_library,
                    auto_add_signatures=auto_add_signatures,
                    conversion_tracker=conversion_tracker,
                )
                function_logic_dags_with_types_added.add(
                    function_logic_dag.graph["name"]
                )
            new_ids, new_skip_stack = expand_node(
                node_id_to_expand=node_id,
                function_logic_dag=function_logic_dag,
                base_dag=dag_to_convert,
                tables_dict=tables_dict,
                signature_definitions=conversion_rules,
                signature_definition_library=signature_definition_library,
                auto_add_signatures=auto_add_signatures,
                conversion_tracker=conversion_tracker,
                separate_tables=separate_tables,
            )
            add_to_queue(new_ids)
            skip_stack.extend(new_skip_stack)
            continue

        # Add parent nodes
        add_to_queue(dag_to_convert.predecessors(node_id))

    nodes_to_lop_off = find_nodes_to_lop_off(
        graph=dag_to_convert, treat_tables_as_dynamic=True
    )
    if len(nodes_to_lop_off) > 0:
        raise ValueError(f"Found nodes that can be lopped off: {nodes_to_lop_off}")
        # for now just stop and see what we have. will work on implementation next.
    # lop them off. Still to do.
    remove_ifs_with_first_node_as_constant(
        G=dag_to_convert, conversion_tracker=conversion_tracker
    )

    if renum_nodes:
        renumber_nodes(dag_to_convert)

    assert validation.is_valid_base_graph(
        dag_to_convert, True
    ), "converted dag is not valid"
    assert validation.is_valid_conversion_rules_dict(
        conversion_rules
    ), "Conversion rules dictioanry is not valid"


def get_ordered_parent_ids(graph, node_id) -> List[int]:
    # Retrieve incoming edges, including edge keys for MultiDiGraph
    edges = list(graph.in_edges(node_id, data="parent_position", keys=True))

    # Sort by "parent_position" attribute
    edges.sort(key=lambda x: x[3])  # x[3] is the parent_position in this case

    # Extract parent node IDs in the sorted order
    parent_ids = [edge[0] for edge in edges]

    return parent_ids


def generate_binomial_dag(function, n, data_type):
    """
    generate a binomial DAG for where all function nodes use the same function
    creates tree with maximum independence possible for given number of inputs
    this diregard for input order means this should only be used for commutative functions
    """
    # Create empty Multi DAG
    G = nx.MultiDiGraph()

    # Add required input nodes
    input_nodes = [i for i in range(n)]
    for node_id in input_nodes:
        G.add_node(
            node_id,
            input_order=node_id,
            node_type="input",
            input_name=f"input_{node_id}",
            data_type=data_type,
        )

    # Initialize list of current nodes (start with input nodes)
    current_nodes = input_nodes.copy()

    # Layer index
    calc_node_id = n

    while len(current_nodes) > 1:
        # List to store new layer nodes
        new_layer_nodes = []

        # Process nodes in pairs
        for i in range(0, len(current_nodes), 2):
            if i + 1 < len(current_nodes):
                # Create a new node for each pair
                G.add_node(
                    calc_node_id,
                    node_type="function",
                    function_name=function,
                    data_type=data_type,
                )
                G.add_edge(current_nodes[i], calc_node_id, parent_position=0)
                G.add_edge(current_nodes[i + 1], calc_node_id, parent_position=1)
                new_layer_nodes.append(calc_node_id)
                calc_node_id += 1
            else:
                # If odd number of nodes, carry the last one to the next layer
                new_layer_nodes.append(current_nodes[i])

        # Update current nodes for the next layer
        current_nodes = new_layer_nodes

    G.graph["input_node_ids"] = input_nodes

    G.nodes[calc_node_id - 1]["output_name"] = "output_placeholder"
    G.nodes[calc_node_id - 1]["output_order"] = 0
    G.graph["output_node_ids"] = [calc_node_id - 1]
    G.graph["max_node_id"] = calc_node_id - 1
    G.graph["name"] = (
        f"binomial_expansion_for_{function}. n: {n}. data_type: {data_type}"
    )
    assert validation.is_valid_graph(G, False), "Graph is not valid"

    return G


def expand_node(
    node_id_to_expand,
    function_logic_dag,
    base_dag,
    tables_dict,
    signature_definitions,
    signature_definition_library,
    auto_add_signatures,
    conversion_tracker,
    separate_tables,
) -> Tuple[List[int], List[int]]:
    """
    The function_logic_dag is the 'definition' of the function represented
    by node_id_to_expand in the base_dag. This function 'expands' the
    node_id_to_expand by replacing it with its calculations, as defined
    by function_logic_dag. It returns a list of the new output node ids.
    """

    def mimic_output_attribs(node_id_to_expand, new_output_node_id, base_dag):
        node_to_expand_attributes = base_dag.nodes[node_id_to_expand]
        new_output_node_attributes = base_dag.nodes[new_output_node_id]

        if (
            "data_type" in node_to_expand_attributes
            and "data_type" in new_output_node_attributes
        ):
            if (
                node_to_expand_attributes["data_type"]
                != new_output_node_attributes["data_type"]
            ):
                if new_output_node_attributes["data_type"] == "Any":
                    new_output_node_attributes["data_type"] = node_to_expand_attributes[
                        "data_type"
                    ]
                else:
                    errs.save_dag_and_raise__node(
                        base_dag,
                        node_id_to_expand,
                        f"Add support for type conversion on function expansion. Node {node_id_to_expand} in tree: {base_dag.graph['name']}.",
                    )

        if "output_order" in node_to_expand_attributes:
            new_output_node_attributes["output_order"] = node_to_expand_attributes[
                "output_order"
            ]

        if "output_name" in node_to_expand_attributes:
            new_output_node_attributes["output_name"] = node_to_expand_attributes[
                "output_name"
            ]

    assert validation.is_valid_base_graph(
        base_dag, False
    ), f"base_dag {base_dag.graph['name']} is not a valid graph. This is the check before expanding node for {function_logic_dag.graph['name']}."

    # Step 1: Retrieve Ordered List of Parent Nodes of Node to Expand
    parents = get_ordered_parent_ids(base_dag, node_id_to_expand)

    if len(parents) != len(function_logic_dag.graph["input_node_ids"]):
        errs.save_2dags_and_raise(
            base_dag,
            function_logic_dag,
            f"Node {node_id_to_expand} in tree: {base_dag.graph['name']} has {len(parents)} parents. Expected {len(function_logic_dag.graph['input_node_ids'])} for function {function_logic_dag.graph['name']}. Dag1 = base_dag; Dag2 = function_logic_dag.",
        )

    id_offset: int = base_dag.graph["max_node_id"]

    # iterate through nodes for steps 2 and 3
    function_logic_nodes = list(function_logic_dag.nodes(data=True))
    for node_id, data in function_logic_nodes:
        # Step 2: Add function nodes and constant nodes from function_logic DAG to Base DAG with ID Adjustments
        if data["node_type"] in ["function", "constant"]:
            new_id: int = node_id + id_offset
            filtered_data = {
                key: value
                for key, value in data.items()
                if key not in ["output_name", "output_order"]
            }
            base_dag.add_node(new_id, **filtered_data)

            # Add edges (dependencies)
            out_edges_to_modify = list(function_logic_dag.out_edges(node_id, data=True))

            for _, target_id, edge_data in out_edges_to_modify:
                new_target_id = target_id + id_offset
                base_dag.add_edge(new_id, new_target_id, **edge_data)

        # Step 3: Rewire Dependencies Based on Input Positions
        elif data["node_type"] == "input":
            input_order = int(data["input_order"])
            if input_order > len(parents) - 1:
                errs.save_dag_and_raise__node(
                    function_logic_dag,
                    node_id,
                    f"function_logic DAG {function_logic_dag.graph['name']} has an input node with an invalid input order.",
                )
            corresponding_parent_node_id = parents[input_order]

            out_edges_to_modify = list(function_logic_dag.out_edges(node_id, data=True))

            # Redirect edges from function_logic_dag's input node to the corresponding
            # parent node of the node_to_expand from the base_dag
            for _, child_id, edge_data in out_edges_to_modify:
                base_dag.add_edge(
                    corresponding_parent_node_id, child_id + id_offset, **edge_data
                )

    # Step 4: Rewire Output Dependencies
    output_node_ids = function_logic_dag.graph["output_node_ids"]
    if len(output_node_ids) == 1:
        output_node_id: int = output_node_ids[0]  # Get the ID of the only output node
        new_output_node_id: int = (
            output_node_id + id_offset
        )  # New ID for the output node

        # for each edge of the original node to expand...
        out_edges_to_modify = list(base_dag.out_edges(node_id_to_expand, data=True))
        for _, child, edge_data in out_edges_to_modify:
            # Add new edge from new output node to the original child
            # xxx check that this **edge_data handles multiDags correctly.
            base_dag.add_edge(new_output_node_id, child, **edge_data)

            # Remove the original edge from the node being expanded to the child
            base_dag.remove_edge(node_id_to_expand, child)

        mimic_output_attribs(node_id_to_expand, new_output_node_id, base_dag)
        new_output_node_ids = [new_output_node_id]
        # If node_id_to_expand is part of the base_dag's output_node_ids
        #   then update the base_dag's output_node_ids to replace it with
        #   the new_output_node_id
        if node_id_to_expand in base_dag.graph["output_node_ids"]:
            base_dag.graph["output_node_ids"] = [
                new_output_node_id if nid == node_id_to_expand else nid
                for nid in base_dag.graph["output_node_ids"]
            ]
    elif len(output_node_ids) > 1:
        # New code begins here.
        new_output_node_ids = []
        nodes_to_remove = []
        edges_to_remove = []
        edges_to_add = []
        for idx, output_node_id in enumerate(
            function_logic_dag.graph["output_node_ids"]
        ):
            new_ouput_is_used = False
            new_output_node_id = output_node_id + id_offset

            # Now process each FUNCTION_ARRAY node (fan_node) depending on its position
            for fan_node_id in base_dag.successors(node_id_to_expand):
                fan_node = base_dag.nodes[fan_node_id]
                if fan_node["function_name"] != "FUNCTION_ARRAY":
                    raise ValueError(f"Node {fan_node} is not a FUNCTION_ARRAY node")

                _, position = (
                    cr.get_parent_function_and_position_for_function_array_node(
                        base_dag, fan_node_id
                    )
                )

                # Check if the position matches the current output node being processed
                if position - 1 == idx:  # Adjusting position to 0-based index
                    new_ouput_is_used = True
                    grandkids = list(base_dag.successors(fan_node_id))
                    for grandkid in grandkids:
                        # Retrieve all edges between fan_node_id and grandkid
                        for edge_key, edge_data in base_dag.get_edge_data(
                            fan_node_id, grandkid
                        ).items():
                            edges_to_remove.append((fan_node_id, grandkid, edge_key))
                            # When copying, no need to call .copy() on edge_data if you're going to modify it anyway
                            edges_to_add.append(
                                (new_output_node_id, grandkid, edge_data, edge_key)
                            )
                    mimic_output_attribs(output_node_id, new_output_node_id, base_dag)
                    nodes_to_remove.append(fan_node_id)

            if new_ouput_is_used:
                new_output_node_ids.append(new_output_node_id)

        for edge in edges_to_remove:
            base_dag.remove_edge(*edge)

        for node in nodes_to_remove:
            base_dag.remove_node(node)

        for edge in edges_to_add:
            source_id, target_id, edge_data, _ = edge
            base_dag.add_edge(source_id, target_id, **edge_data)

    else:
        raise ValueError(
            f"Function logic DAG {function_logic_dag.graph['name']} has an invalid number of outputs."
        )

    base_dag.remove_node(node_id_to_expand)

    skip_stack = remove_all_non_output_sink_nodes(base_dag)

    base_dag.graph["max_node_id"] = max(base_dag.nodes())

    # cleanup new nodes
    new_nodes = [node_id for node_id in base_dag.nodes() if node_id > id_offset]

    # separate out tables for new nodes
    if separate_tables:
        g_tables.separate_named_tables(base_dag, new_nodes, tables_dict)

    # set data types for new nodes
    sorted_new_nodes = topo_sort_subgraph(base_dag, new_nodes)
    update_dag_with_data_types(
        base_dag,
        sorted_new_nodes,
        signature_definitions,
        signature_definition_library,
        auto_add_signatures,
        conversion_tracker,
    )

    # Final Steps: Update Graph Attributes and Check Integrity
    assert validation.is_valid_base_graph(
        base_dag, False
    ), f"base_dag {base_dag.graph['name']} is not a valid graph. Check after expanding node for {function_logic_dag.graph['name']}."

    return new_output_node_ids, skip_stack


def remove_all_non_output_sink_nodes(G):
    removed: List[int] = []
    # Initialize a queue with initial non-output sink nodes
    initial_sink_nodes = [
        node
        for node, out_degree in G.out_degree()
        if out_degree == 0 and "output_name" not in G.nodes[node]
    ]
    queue = deque(initial_sink_nodes)

    while queue:
        node = queue.popleft()
        # Before removal, get predecessors to check if they become new sinks after removal
        predecessors = list(G.predecessors(node))
        G.remove_node(node)
        removed.append(node)
        for pred in predecessors:
            # If a predecessor is now a sink and does not have 'output_name', add it to the queue
            if G.out_degree(pred) == 0 and "output_name" not in G.nodes[pred]:
                queue.append(pred)

    return removed


def dict_of_matching_node_ids(
    base_dag, transform_dag, base_node_id, transform_node_id
) -> Dict[str, int]:
    def node_match(bs_node_attribs, tr_node_attribs):
        # bs = base; tr = transform
        # Check if the data types are the same. Note often tr_node will not have a data_type in which case this will return true.
        if not cr.match_type(
            bs_node_attribs.get("data_type"),
            tr_node_attribs.get("data_type"),
            False,
            False,
        ):
            return False

        if (
            bs_node_attribs["node_type"] == "input"
        ):  # but what if they are both inupts but different types?
            return tr_node_attribs["node_type"] == "input"

        if "output_name" in bs_node_attribs:
            return (
                "output_name" in tr_node_attribs
            )  # again let's check if the type is the same

        if tr_node_attribs["node_type"] == "input":
            return True

        if bs_node_attribs["node_type"] != tr_node_attribs["node_type"]:
            return False

        # For FunctionNodes, compare based on the 'Name' attribute
        if bs_node_attribs["node_type"] == "function":
            return bs_node_attribs["function_name"] == tr_node_attribs["function_name"]

        # For ConstantNodes, compare based on value and data_type
        if bs_node_attribs["node_type"] == "constant":
            return (
                bs_node_attribs["value"] == tr_node_attribs["value"]
                and bs_node_attribs["data_type"] == tr_node_attribs["data_type"]
            )  # consider coming back here and making number matches less sensitive.

        raise ValueError(
            f"Unsupported node comparison: {bs_node_attribs['ID']}, {tr_node_attribs['ID']}"
        )

    def dfs(base_node_id, transform_node_id, mapping):
        # Check if the current nodes match
        if not node_match(
            base_dag.nodes[base_node_id], transform_dag.nodes[transform_node_id]
        ):
            return False

        # Create keys for mapping
        base_key = f"b-{base_node_id}"
        transform_key = f"t-{transform_node_id}"

        # Check if either node is already in the mapping
        if base_key in mapping or transform_key in mapping:
            # Check if they are mapped to each other
            if (
                mapping.get(base_key) != transform_node_id
                or mapping.get(transform_key) != base_node_id
            ):
                return False

        # Add the current node pair to the mapping
        mapping[f"b-{base_node_id}"] = transform_node_id
        mapping[f"t-{transform_node_id}"] = base_node_id

        # Get predecessors of both nodes sorted by parent_position
        base_predecessors = [
            source
            for source, _, data in sorted(
                base_dag.in_edges(base_node_id, data=True),
                key=lambda edge: edge[2]["parent_position"],
            )
        ]
        transform_predecessors = [
            source
            for source, _, data in sorted(
                transform_dag.in_edges(transform_node_id, data=True),
                key=lambda edge: edge[2]["parent_position"],
            )
        ]
        if (
            len(transform_predecessors) > 0
        ):  # would equal 0 if the transform node is an input node for the transform. in that case skip the comparison.
            # If the number of predecessors is different, the pattern doesn't match
            if len(base_predecessors) != len(transform_predecessors):
                return False

            # Recursively check predecessors
            for base_pred_id, transform_pred_id in zip(
                base_predecessors, transform_predecessors
            ):
                if not dfs(base_pred_id, transform_pred_id, mapping):
                    return False

        return True

    mapping = {}
    match_found = dfs(base_node_id, transform_node_id, mapping)

    if match_found:
        return mapping
    else:
        return {}


def remove_edge(G, node_id, parent_node_id, parent_position):
    """
    Remove an edge between two nodes based on the parent_position attribute.

    Parameters:
    - G: The graph (nx.MultiDiGraph)
    - node_id: The ID of the node
    - parent_node_id: The ID second node--must be a predecessor of the first node
    - parent_position: The parent_position attribute value to match for removal
    """
    # Iterate over all edges between the specified nodes
    for u, _, key, attr in G.in_edges(node_id, data=True, keys=True):
        if u == parent_node_id and attr.get("parent_position") == parent_position:
            G.remove_edge(u, node_id, key)
            break


def remove_ifs_with_first_node_as_constant(G: nx.MultiDiGraph, conversion_tracker):
    if not nx.is_weakly_connected(G):
        raise ValueError("Graph is not weakly connected")

    # Step 1: Gather information
    node_modifications = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("function_name") == "IF":
            parents = get_ordered_parent_ids(G, node_id)
            if len(parents) != 3:
                raise ValueError(
                    f"Expected 3 parents for IF node {node_id}, got {len(parents)}"
                )
            if G.nodes[parents[0]]["node_type"] == "constant":
                first_parent = parents[0]
                is_true = bool(
                    convert_to_python_type(
                        G.nodes[first_parent]["value"],
                        G.nodes[first_parent]["data_type"],
                    )
                )
                parent_id_to_use = parents[1] if is_true else parents[2]
                parent_id_not_used = parents[2] if is_true else parents[1]
                node_modifications.append(
                    (node_id, first_parent, parent_id_to_use, parent_id_not_used)
                )

    # Step 2: Modify the graph based on the gathered information. This avoids looping over the graph
    # while we remove nodes and edges
    for (
        node_id,
        first_parent,
        parent_id_to_use,
        parent_id_not_used,
    ) in node_modifications:
        remove_edge(G, node_id, first_parent, 0)
        if G.out_degree(first_parent) == 0:
            G.remove_node(first_parent)
        remove_node_rewire_children_to_parent(G, node_id, parent_id_to_use)

        # Record change in conversion tracker
        ct.update_conversion_tracker_event(
            conversion_tracker, "remove_if_with_first_node_as_constant"
        )


def remove_node_and_ancestors_safe(G, start_node):
    # Initialize a queue with the start node
    queue = [start_node]
    # Initialize a set to track all nodes to potentially remove
    sub_graph_nodes = set([start_node])

    while queue:
        current_node = queue.pop(0)  # Dequeue the next node
        # Check if current_node has successors outside sub_graph_nodes
        if any(
            successor not in sub_graph_nodes for successor in G.successors(current_node)
        ):
            continue  # Skip removal if there are external successors

        # For nodes with no external successors, check their predecessors
        for pred in G.predecessors(current_node):
            if pred not in sub_graph_nodes:
                sub_graph_nodes.add(pred)
                queue.append(pred)  # Enqueue predecessor for further checks

    # Remove all identified nodes from the graph
    for node in sub_graph_nodes:
        G.remove_node(node)


def remove_node_rewire_children_to_parent(G, node_id, parent_node_id):
    """
    Remove node_id from the graph, connecting parent_node_id directly to node_id's
    successors (parent node_id's grandkids through node_id). Preserve the parent_position
    attributes of the edges from node_id to its successors.

    Parameters:
    - G: A directed graph (nx.MultiDiGraph).
    - node_id: The ID of the node to be removed, whose outgoing edges are to be transferred.
    - parent_node_id: The ID of the node to receive the transferred edges, and a parent of node_id
    """
    # Correctly collect all outgoing edges from node_id with their parent_position
    outgoing_edges = [
        (node_id, succ, attr["parent_position"])
        for succ, attr in G[node_id].items()
        for _, attr in G[node_id][succ].items()
    ]

    # Remove the node_id, which will also eliminate all its connections
    G.remove_node(node_id)

    # Add new edges from node_id_new to the successors of the original node_id
    for _, successor, parent_position in outgoing_edges:
        G.add_edge(parent_node_id, successor, parent_position=parent_position)


def find_nodes_to_lop_off(graph: nx.MultiDiGraph, treat_tables_as_dynamic) -> set:
    """
    Identify function nodes that represent a transition from static (fixed) to dynamic computation.

    :param graph: A directed multigraph where nodes represent computations or data and edges represent dependencies.
    :param treat_tables_as_dynamic: If True, treat table arrays as dynamic inputs; otherwise, they are considered fixed.
    :return: A set of nodes that are function nodes marking the critical transition point from fixed to dynamic.
    """
    # Initialize sets for fixed, dynamic, and transition nodes
    # all nodes will be added to either fixed or dynamic.
    fixed, dynamic, transition = set(), set(), set()

    # Topological sort of the graph to process nodes in linearized order
    sorted_nodes = list(nx.topological_sort(graph))

    for node in sorted_nodes:
        node_type = graph.nodes[node]["node_type"]

        # Directly categorize constant and input nodes
        if node_type == "constant":
            fixed.add(node)
        elif node_type == "input":
            dynamic.add(node)
        elif node_type == "table_array":
            if treat_tables_as_dynamic:
                dynamic.add(node)
            else:
                fixed.add(node)

        # Process function nodes based on the status of their immediate predecessors
        elif node_type == "function":
            parents = graph.predecessors(node)
            if all(parent in fixed for parent in parents):
                fixed.add(node)
            else:
                # If any parent is dynamic, the node is dynamic and parents that are functions become transitions
                dynamic.add(node)
                for parent in parents:
                    if (
                        parent in fixed
                        and graph.nodes[parent]["node_type"] == "function"
                    ):
                        transition.add(parent)

    return transition


def main():
    pass
