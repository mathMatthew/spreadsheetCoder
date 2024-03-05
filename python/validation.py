import networkx as nx


def is_valid_graph(G, require_types: bool) -> bool:
    def nodes_have_attributes(node_ids, desired_attributes):
        for node_id in node_ids:
            attribs = G.nodes[node_id]
            for attr in desired_attributes:
                if attr not in attribs:
                    return False
        return True

    if not nx.is_directed_acyclic_graph(G):
        return False

    if not nx.is_weakly_connected(G):
        return False

    # all source nodes are input nodes or constants.
    source_node_ids = [node for node, in_degree in G.in_degree() if in_degree == 0]
    for node_id in source_node_ids:
        if G.nodes[node_id]["node_type"] not in {"constant", "input", "table_array"}:
            return False

    # validate input_node_ids
    input_node_ids = [
        node_id
        for node_id, attrs in G.nodes(data=True)
        if attrs.get("node_type") == "input"
        or "input_order" in attrs
        or "input_name" in attrs
    ]
    if not nodes_have_attributes(input_node_ids, ["input_name", "input_order"]):
        return False
    if not all(
        G.nodes[node_id].get("node_type") == "input" for node_id in input_node_ids
    ):
        return False
    sorted_inputs = sorted(
        input_node_ids, key=lambda node_id: G.nodes[node_id]["input_order"]
    )
    if G.graph["input_node_ids"] != sorted_inputs:
        return False

    # all sink nodes are output nodes
    sink_node_ids = [node for node, out_degree in G.out_degree() if out_degree == 0]
    for node_id in sink_node_ids:
        if "output_name" in G.nodes[node_id]:
            continue
        return False

    # validate output_node_ids
    output_node_ids = [
        node
        for node, data in G.nodes(data=True)
        if "output_name" in data or "output_order" in data
    ]
    if not nodes_have_attributes(output_node_ids, ["output_name", "output_order"]):
        return False
    sorted_outputs = sorted(
        output_node_ids, key=lambda node_id: G.nodes[node_id]["output_order"]
    )
    if G.graph["output_node_ids"] != sorted_outputs:
        return False

    # all edges have attribute 'parent_position'
    for _, _, attrs in G.edges(data=True):
        if "parent_position" not in attrs:
            return False

    # validate max_node_id
    if G.graph["max_node_id"] != max(G.nodes()):
        return False

    # all nodes have node_type if node_type is required
    # if a node has multiple outputs variable will be named node_types instead. validate that
    # it can have multiple outputs using that function
    if require_types:
        for node in G.nodes():
            if "data_type" not in G.nodes[node]:
                # validate that if a node has multiple outputs than all of its successors are function array nodes
                if "data_types" in G.nodes[node]:
                    if not node_can_have_multiple_outputs(G, node):
                        return False
                else:
                    return False

    return True


def node_can_have_multiple_outputs(G, node_id):
    """
    Returns true if all dependents of a node are function arrays
    Note if this is an output node, it will return true by design
    as this is the other condition under which a node can have multiple outputs
    """
    condition_met = True
    for successor_id in G.successors(node_id):
        if G.nodes[successor_id]["node_type"] == "function":
            if G.nodes[successor_id]["function_name"] == "FUNCTION_ARRAY":
                continue
        condition_met = False
        break
    return condition_met


def is_valid_transform(transform_logic_dag, filename, name):
    if name != filename.split(".")[0].upper():
        return False

    if not is_valid_graph(transform_logic_dag, False):
        return False

    outputs = transform_logic_dag.graph["output_node_ids"]

    if len(outputs) < 2:
        return False

    if transform_logic_dag.nodes[outputs[0]]["output_name"].upper() != "FROM":
        return False

    if transform_logic_dag.nodes[outputs[1]]["output_name"].upper() != "TO":
        return False

    return True


def is_valid_base_graph(base_dag) -> bool:
    return is_valid_graph(base_dag, False)


def is_valid_logic_function(tree, filename, name):
    if name != filename.split(".")[0].upper():
        return False

    if not is_valid_graph(tree, False):
        return False

    return True


def is_valid_signature_definition_dict(
    conversion_rules, allow_multiple_outputs, is_library
) -> bool:
    # a signature definition dictionary must have a 'signatures' key with a list of signatures
    # each signature must have 'inputs' and 'outputs' keys
    # each input and output must be a list of data types
    # unless is_library is true, cannot have more than one signature with the same inputs.

    if not "signatures" in conversion_rules:
        return False
    conversion_rules = conversion_rules["signatures"]

    seen_function_inputs = {}

    for func_name, signatures in conversion_rules.items():
        for signature in signatures:
            # Unless a library, don't allow dupes of function_name and input data types
            # though this doesnt use cr.match_input_signature(... "exact") it is doing the same thing. This comment is important for identifying cases where this function belongs.
            if not is_library:
                func_input_signature = (func_name, tuple(signature["inputs"]))
                if func_input_signature in seen_function_inputs:
                    print(
                        f"Duplicate function signature detected for {func_name} with inputs {', '.join(signature['inputs'])}."
                    )
                    return False
                seen_function_inputs[func_input_signature] = True

            for input_type in signature["inputs"]:
                if not valid_data_type(input_type, False):
                    print(
                        f"Function signature {func_name} has input with invalid data_type: {input_type}."
                    )
                    return False

            outputs = signature["outputs"]
            if not allow_multiple_outputs and len(outputs) > 1:
                print(
                    f"Function signature {func_name} has {len(outputs)} outputs. Add support for multiple outputs to use"
                )
                return False

            for output_type in signature["outputs"]:
                if not valid_data_type(output_type, False):
                    print(
                        f"Function signature {func_name} has output with invalid data_type: {output_type}."
                    )
                    return False

    return True


def is_valid_conversion_rules_dict(conversion_rules_dict) -> bool:
    # a conversion rules dictionary is a signture definition dictionary plus
    # the other rules needed for to convert the dag to the target language
    if not is_valid_signature_definition_dict(conversion_rules_dict, True, False):
        return False

    for func_name, signatures in conversion_rules_dict["signatures"].items():
        for signature in signatures:
            # some signatures in the conversion dictionary don't provide any information on how to convert them
            # this can be valid if the function is only used as an intermediate step. In that case they need to be
            # labeled "no_code".
            if not (
                signature.get("operator")
                or signature.get("code_before")
                or signature.get("template")
            ):
                if not signature.get("no_code"):
                    return False
            else:
                # a signature can use a template or the before,operator, after stuff, but not both.
                if signature.get("template"):
                    if (
                        signature.get("operator")
                        or signature.get("code_before")
                        or signature.get("code_after")
                        or signature.get(
                            "add_functions"
                        )  # if a template requires a function that fact is noted in the templates section, not in the signature.
                    ):
                        return False
                    # if a signature has a template name, make sure it matches the templates in the conversion_rules_dict
                    if not conversion_rules_dict.get("templates", {}).get(
                        signature["template"]
                    ):
                        return False
                if signature.get("add_function"):
                    for add_function in signature["add_functions"]:
                        # for all functions to add, make sure their is a matching functions in the conversion_rules_dict
                        if not conversion_rules_dict.get("functions", {}).get(
                            add_function
                        ):
                            return False

            if "template" in signature:
                if not conversion_rules_dict.get("templates", {}).get(
                    signature["template"]
                ):
                    return False

    return True


def is_valid_conversion_tracker(conversion_tracker):
    # Check if the main structure is a dictionary
    if not isinstance(conversion_tracker, dict):
        print("Conversion tracker is not a dictionary.")
        return False

    required_keys = [
        "events",
        "signatures",
        "transforms",
        "expanded_functions",
        "binomial_expansions",
        "templates_used",
    ]

    # Check for required keys
    if not all(key in conversion_tracker for key in required_keys):
        print("Conversion tracker is missing one or more required keys")
        return False

    # Validate 'events' sub-dictionary
    events = conversion_tracker["events"]
    if not isinstance(events, dict):
        print("Conversion tracker 'events' sub-dictionary is not a dictionary")
        return False
    for event_data in events.values():
        if not isinstance(event_data, dict):
            print(
                "Conversion tracker 'events' sub-dictionary has one or more item that is not a dictionary"
            )
            return False
        if "usage_count" not in event_data or not isinstance(
            event_data["usage_count"], int
        ):
            print(
                "Conversion tracker 'events' sub-dictionary has one or more item that is missing the 'usage_count' key or the usage count value is not an integer"
            )
            return False

    # Validate 'signatures' sub-dictionary
    signatures = conversion_tracker["signatures"]
    if not isinstance(signatures, dict):
        print("Conversion tracker 'signatures' sub-dictionary is not a dictionary")
        return False
    for sig_list in signatures.values():
        if not isinstance(sig_list, list):
            print(
                "Conversion tracker 'signatures' sub-dictionary has one or more item where value is not a list"
            )
            return False
        for sig_data in sig_list:
            if "inputs" not in sig_data or "outputs" not in sig_data:
                print(
                    "Conversion tracker 'signatures' sub-dictionary has one or more item that is missing the 'inputs' or 'outputs' key"
                )
                return False
            if not isinstance(sig_data["inputs"], list) or not isinstance(
                sig_data["outputs"], list
            ):
                print(
                    "Conversion tracker 'signatures' sub-dictionary has one or more item that has the 'inputs' or 'outputs' key that is not a list"
                )
                return False

    # Validate 'transforms' sub-dictionary
    if not isinstance(conversion_tracker["transforms"], dict):
        print("Conversion tracker 'transforms' sub-dictionary is not a dictionary")
        return False
    if not all(
        isinstance(value, int) for value in conversion_tracker["transforms"].values()
    ):
        print(
            "Conversion tracker 'transforms' sub-dictionary has one or more item where the value is not an integer"
        )
        return False

    # Validate 'expanded_functions' sub-dictionary
    if not isinstance(conversion_tracker["expanded_functions"], dict):
        print(
            "Conversion tracker 'expanded_functions' sub-dictionary is not a dictionary"
        )
        return False
    if not all(
        isinstance(value, int)
        for value in conversion_tracker["expanded_functions"].values()
    ):
        print(
            "Conversion tracker 'expanded_functions' sub-dictionary has one or more item where the value is not an integer"
        )
        return False

    # If all checks pass
    return True


def valid_data_type(data_type, is_strict):
    valid_data_types = ["Text", "Number", "Boolean", "Date"]
    if not is_strict:
        valid_data_types.append("Any")
    if data_type in valid_data_types:
        return True
    elif data_type.startswith("Multiple[ARRAY[") and data_type.endswith("]]"):
        inner_type = data_type[15:-2]
        return inner_type in valid_data_types
    elif data_type.startswith("Multiple[") and data_type.endswith("]"):
        inner_type = data_type[9:-1]
        return inner_type in valid_data_types
    elif data_type.startswith("ARRAY[") and data_type.endswith("]"):
        inner_type = data_type[6:-1]
        return inner_type in valid_data_types
    elif data_type.startswith("TABLE_COLUMN[") and data_type.endswith("]"):
        inner_type = data_type[13:-1]
        return inner_type in valid_data_types
    else:
        return False
