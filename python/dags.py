from typing import Any, Dict, Tuple, List
import networkx as nx
from collections import deque
import os, json

import validation, errs, setup
import conv_tracker as ct
import convert_xml as cxml
import signatures as sigs

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

    # add names for named nodes.
    for named_node in lxml_tree.find("NamedNodes"):
        node_attribs = G.nodes[int(named_node.attrib["node_id"])]
        if "node_name_type" not in node_attribs:
            node_attribs["node_name_type"] = named_node.attrib["node_name_type"]
            node_attribs["node_name"] = named_node.attrib["node_name"]
            continue
        # if it already has a name of the same type, raise an error.
        if named_node.attrib["node_name_type"] == node_attribs["node_name_type"]:
            cxml.save_xml_and_raise(
                lxml_tree,
                f"Duplicate node_name_type detected. Node_id: {named_node.attrib['node_id']}, node_name_type: {node_attribs['node_name_type']}",
            )
        # alias names take priority. if it already has an alias name skip it.
        if node_attribs["node_name_type"] == "alias":
            continue
        # if it has an address name then this is an alias name (only 2 options)
        # so overwrite address name with alias name
        elif node_attribs["node_name_type"] == "address":
            node_attribs["node_name_type"] = named_node.attrib["node_name_type"]
            node_attribs["node_name"] = named_node.attrib["node_name"]
        else:
            raise ValueError(
                f"Unknown node_name_type: {node_attribs['node_name_type']}"
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
    conversion_tracker,
):
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."
    assert validation.is_valid_base_graph(
        base_dag
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
    new_id = expand_node(node_id_to_replace, to_dag, base_dag)
    assert validation.is_valid_base_graph(
        base_dag
    ), f"base_dag is not valid after transform {transform_from_to_dag.graph['name']}"
    return new_id


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


def convert_to_binomial(G, node_id, comm_func_dict, conversion_tracker):
    """
    Convert the given node to a graph of binomial functions, ensuring that all input data types are the same.
    Parameters:
    - G: the graph
    - node_id: the ID of the node to be converted
    - comm_func_dict: the dictionary entry for the commutative function
    - conversion_tracker: the conversion tracker
    """
    parent_data_types = sigs.get_parent_data_types(G, node_id)
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
    new_id = expand_node(node_id, new_dag, G)
    return new_id


def update_dag_with_data_types(
    G: nx.MultiDiGraph,
    conversion_signatures: Dict,
    library_func_sigs: Dict,
    auto_add: bool,
    conversion_tracker: Dict,
):
    assert nx.is_directed_acyclic_graph(G), f'{G.graph["name"]} is not a DAG'
    assert validation.is_valid_conversion_fn_sig_dict(
        conversion_signatures
    ), "signature is not valid"
    assert validation.is_valid_fn_sig_dict(library_func_sigs), "signature is not valid"

    topo_sorted_nodes = list(nx.topological_sort(G))

    for node_id in topo_sorted_nodes:
        if "data_type" in G.nodes[node_id]:
            continue
        if "function_name" not in G.nodes[node_id]:
            errs.save_dag_and_raise_node(
                G,
                node_id,
                f"Node id: {node_id} does not have data_type but is not a function.",
            )
            return
        G.nodes[node_id]["data_type"] = sigs.get_data_type(
            G,
            node_id,
            conversion_signatures,
            library_func_sigs,
            auto_add,
            conversion_tracker,
        )


def mark_nodes_for_caching(
    G,
    usage_count_threshold,
    branching_threshold,
    complexity_threshold,
    all_array_nodes,
    all_outputs,
    conversion_func_sigs,
):
    """
    Marks nodes for caching based on usage count and computational complexity.

    Parameters:
    G (nx.DiGraph): The directed graph.
    usage_count_threshold (int): The threshold for usage count above which nodes are cached.
    complexity_threshold (int): The threshold for computational complexity above which nodes are cached.
    """

    def calc_branching_depth_and_calc_complexity(G, node_id, memo) -> Tuple[int, int]:
        if node_id in memo:
            return memo[node_id]
        node_attribs = G.nodes[node_id]
        if node_attribs["cache"] == True:
            memo[node_id] = 0, 0
            return memo[node_id]
        if node_attribs["node_type"] != "function":
            memo[node_id] = 0, 0
            return memo[node_id]

        if node_attribs["function_name"] == "if":
            parent_ids = get_ordered_parent_ids(G, node_id)
            condition_id, if_true_id, if_false_id = (
                parent_ids[0],
                parent_ids[1],
                parent_ids[2],
            )

            # Calculating branch depths and complexities
            (
                condition_branch_depth,
                condition_complexity,
            ) = calc_branching_depth_and_calc_complexity(G, condition_id, memo)
            (
                if_true_branch_depth,
                if_true_complexity,
            ) = calc_branching_depth_and_calc_complexity(G, if_true_id, memo)
            (
                if_false_branch_depth,
                if_false_complexity,
            ) = calc_branching_depth_and_calc_complexity(G, if_false_id, memo)

            # Computing the combined branch depth and complexity
            branch_depth = 1 + max(
                condition_branch_depth, if_true_branch_depth, if_false_branch_depth
            )
            calculation_complexity = (
                1 + condition_complexity + max(if_true_complexity, if_false_complexity)
            )
        else:
            # for non-if
            # branch complexity = max of parents
            # calculation complexity = sum of parents
            parent_complexities: list[tuple[int, int]] = [
                calc_branching_depth_and_calc_complexity(G, source, memo)
                for source, _ in G.in_edges(node_id)
            ]
            branch_depth = max(depth for depth, _ in parent_complexities)
            calculation_complexity = sum(
                complexity for _, complexity in parent_complexities
            )

        memo[node_id] = branch_depth, calculation_complexity
        return branch_depth, calculation_complexity

    if all_array_nodes:
        for node_id in G.nodes:
            if G.nodes[node_id].get("function_name", "").upper() == "ARRAY":
                G.nodes[node_id]["cache"] = True

    # mark all outputs for caching, if applicable
    if all_outputs:
        for node_id in G.graph["output_node_ids"]:
            G.nodes[node_id]["cache"] = True

    # mark nodes for caching for nodes which functions require caching
    for node_id in G.nodes:
        if G.nodes[node_id]["node_type"] == "function":
            function_signature = sigs.match_signature(G, node_id, conversion_func_sigs)
            if function_signature and function_signature.get("requires_cache", False):
                G.nodes[node_id]["cache"] = True

    # First, mark nodes for caching based on usage count
    for node_id in G.nodes:
        if G.nodes[node_id].get("cache"):
            continue
        usage_count = 0
        for _, target in G.in_edges(node_id):
            usage_count += 1
        G.nodes[node_id]["cache"] = (
            True if usage_count > usage_count_threshold else False
        )

    # topo sort of graph will mean that when cPerform a topological sort of the graph
    sorted_nodes = list(nx.topological_sort(G))

    memo: dict[int, Tuple[int, int]] = {}  # Initialize the memoization dictionary

    for node_id in sorted_nodes:
        branching_depth, calc_complexity = calc_branching_depth_and_calc_complexity(
            G, node_id, memo
        )
        if (
            branching_depth > branching_threshold
            or calc_complexity > complexity_threshold
        ):
            G.nodes[node_id]["cache"] = True


def convert_graph(
    dag_to_convert: nx.MultiDiGraph,
    function_logic_dags: Dict[str, nx.MultiDiGraph],
    transforms_from_to: Dict[str, nx.MultiDiGraph],
    transforms_protect: Dict[str, nx.MultiDiGraph],
    conversion_signatures: Dict[str, Any],
    library_func_sigs: Dict[str, Any],
    auto_add_signatures: bool,
    conversion_tracker: Dict[str, Any],
    renum_nodes=True,
) -> None:
    """
    Converts a directed acyclic graph (DAG) (the dag_to_convert) into a DAG transformed
    by applying the defined transforms and the function_logic_dags.


    """
    assert validation.is_valid_base_graph(dag_to_convert), "dag is not valid"
    assert validation.is_valid_conversion_fn_sig_dict(
        conversion_signatures
    ), "signature is not valid"
    assert validation.is_valid_fn_sig_dict(library_func_sigs), "signature is not valid"

    func_logic_sigs = sigs.create_signature_dictionary(function_logic_dags)
    sigs.add_signatures_to_library(
        func_logic_sigs, library_func_sigs, "function_logic_dag"
    )

    update_dag_with_data_types(
        dag_to_convert,
        conversion_signatures,
        library_func_sigs,
        auto_add_signatures,
        conversion_tracker,
    )
    assert validation.is_valid_base_graph(dag_to_convert), "dag is not valid"

    def add_to_queue(node_id):
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
    seen = set(stack)  # Set to keep track of nodes already in the stack

    array1_nodes_that_were_checked_for_arrays = set()
    function_logic_dags_with_types_added = set()

    while stack:
        assert validation.is_valid_base_graph(
            dag_to_convert
        ), "converted dag is not valid"
        node_id = stack.pop()

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
        commutative_functions_to_convert_to_binomial = conversion_signatures.get(
            "commutative_functions_to_convert_to_binomial", {}
        )
        if function_name in commutative_functions_to_convert_to_binomial:
            comm_func_dict = commutative_functions_to_convert_to_binomial[function_name]
            new_id = convert_to_binomial(
                dag_to_convert, node_id, comm_func_dict, conversion_tracker
            )
            add_to_queue(new_id)
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
                        new_id: int = transform_from_to(
                            base_dag=dag_to_convert,
                            transform_from_to_dag=transform_logic_dag,
                            node_id_to_replace=node_id,
                            match_mapping=match_mapping,
                            conversion_tracker=conversion_tracker,
                        )
                        add_to_queue(new_id)
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
                    function_logic_dag,
                    conversion_signatures,
                    library_func_sigs,
                    auto_add_signatures,
                    conversion_tracker,
                )
                function_logic_dags_with_types_added.add(
                    function_logic_dag.graph["name"]
                )
            new_id = expand_node(node_id, function_logic_dag, dag_to_convert)
            add_to_queue(new_id)
            continue

        # Add parent nodes
        for parent_id in dag_to_convert.predecessors(node_id):
            add_to_queue(parent_id)

    # because we don't insist that transforms have data types, we need to run one more time.
    update_dag_with_data_types(
        dag_to_convert,
        conversion_signatures,
        library_func_sigs,
        auto_add_signatures,
        conversion_tracker,
    )
    if renum_nodes:
        renumber_nodes(dag_to_convert)
    assert validation.is_valid_base_graph(dag_to_convert), "converted dag is not valid"


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
    if not validation.is_valid_graph(G, False):
        raise ValueError("Graph is not valid")

    return G


def xml_to_graph(
    base_dag_xml_file, working_directory, conversion_tracker, override_defaults
) -> nx.MultiDiGraph:
    data_dict = setup.get_standard_settings(base_dag_xml_file, working_directory)
    data_dict.update(override_defaults)
    # convert graph is the core logic. ...
    convert_graph(
        dag_to_convert=data_dict["base_dag_graph"],
        function_logic_dags=data_dict["function_logic_dags"],
        transforms_from_to=data_dict["transforms_from_to"],
        transforms_protect=data_dict["transforms_protect"],
        conversion_signatures=data_dict["conversion_func_sigs"],
        library_func_sigs=data_dict["library_func_sigs"],
        auto_add_signatures=data_dict["auto_add_signatures"],
        conversion_tracker=conversion_tracker,
    )

    return data_dict["base_dag_graph"]


def expand_node(node_id_to_expand, function_logic_dag, base_dag) -> int:
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
                    errs.save_dag_and_raise_node(
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

    assert validation.is_valid_graph(
        base_dag, False
    ), f"base_dag {base_dag.graph['name']} is not a valid graph. check before expanding node for {function_logic_dag.graph['name']}."

    # Step 1: Retrieve Ordered List of Parent Nodes of Node to Expand
    parents = get_ordered_parent_ids(base_dag, node_id_to_expand)

    if len(parents) != len(function_logic_dag.graph["input_node_ids"]):
        errs.save_2dags_and_raise(
            base_dag,
            function_logic_dag,
            f"Node {node_id_to_expand} in tree: {base_dag.graph['name']} has {len(parents)} parents. Expected {len(function_logic_dag.graph['input_node_ids'])} for function {function_logic_dag.graph['name']}. Dag1 = base_dag; Dag2 = function_logic_dag.",
        )

    id_offset: int = base_dag.graph["max_node_id"]

    counter = 0
    # iterate through nodes for steps 2 and 3
    function_logic_nodes = list(function_logic_dag.nodes(data=True))
    for node_id, data in function_logic_nodes:
        # Step 2: Add function nodes and constant nodes from function_logic DAG to Base DAG with ID Adjustments
        if data["node_type"] in ["function", "constant"]:
            new_id: int = node_id + id_offset
            if new_id == 81:
                print("")
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
                errs.save_dag_and_raise_node(
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
    if len(output_node_ids) > 1:
        errs.save_dag_and_raise_message(
            function_logic_dag,
            f"function_logic DAG {function_logic_dag.graph['name']} has multiple output nodes. Add support for multiple output nodes to use this function_logic",
        )
    output_node_id: int = output_node_ids[0]  # Get the ID of the only output node
    new_output_node_id: int = output_node_id + id_offset  # New ID for the output node

    # for each edge of the original node to expand...
    out_edges_to_modify = list(base_dag.out_edges(node_id_to_expand, data=True))
    for _, child, edge_data in out_edges_to_modify:
        # Add new edge from new output node to the original child
        base_dag.add_edge(new_output_node_id, child, **edge_data)

        # Remove the original edge from the node being expanded to the child
        base_dag.remove_edge(node_id_to_expand, child)

    mimic_output_attribs(node_id_to_expand, new_output_node_id, base_dag)

    if node_id_to_expand in base_dag.graph["output_node_ids"]:
        base_dag.graph["output_node_ids"].remove(node_id_to_expand)
        base_dag.graph["output_node_ids"].append(new_output_node_id)

    base_dag.remove_node(node_id_to_expand)

    # Final Steps: Update Graph Attributes and Check Integrity
    base_dag.graph["max_node_id"] = id_offset + function_logic_dag.graph["max_node_id"]

    assert validation.is_valid_graph(
        base_dag, False
    ), f"base_dag {base_dag.graph['name']} is not a valid graph. Check after expanding node for {function_logic_dag.graph['name']}."

    return new_output_node_id


def dict_of_matching_node_ids(
    base_dag, transform_dag, base_node_id, transform_node_id
) -> Dict[str, int]:
    def node_match(bs_node_attribs, tr_node_attribs):
        # bs = base; tr = transform
        # Check if the data types are the same. Note often tr_node will not have a data_type in which case this will return true.
        if not sigs.match_type(
            bs_node_attribs.get("data_type"), tr_node_attribs.get("data_type"), False
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


def main():
    base_dag_xml_file = "MultipleMatchInexactNumeric.xml"
    working_directory = "../../../OneDrive/Documents/myDocs/sc_v2_data"
    paths_dict = setup.get_standard_paths(base_dag_xml_file, working_directory)

    conversion_tracker: Dict[str, Any] = {"func_sigs": {}, "events": {}}

    base_dag_graph = xml_to_graph(
        base_dag_xml_file, working_directory, conversion_tracker, {}
    )
    data = nx.node_link_data(base_dag_graph)

    with open(paths_dict["json_output_file"], "w") as file:
        json.dump(data, file, indent=2)

    with open(paths_dict["conversion_tracker_file"], "w") as file:
        json.dump(conversion_tracker, file, indent=2)