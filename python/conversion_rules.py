import json, os
import networkx as nx
from typing import Any, Dict, Tuple, List, Optional
from functools import partial

import validation, dags, ui, errs
import conv_tracker as ct


def initialize_conversion_rules() -> Dict[str, Any]:
    return {
        "signatures": {},
        "templates": {},
        "commutative_functions_to_convert_to_binomial": {},
        "functions": {},
        "transforms": {},
        "function_logic_dags": {},
    }


def update_conversion_rules(lower_priority_rules, higher_priority_rules):
    # Update the signatures with more specific logic for matching and updating
    for func_name, higher_signatures in higher_priority_rules["signatures"].items():
        if func_name not in lower_priority_rules["signatures"]:
            lower_priority_rules["signatures"][func_name] = higher_signatures
        else:
            for higher_signature in higher_signatures:
                matched = False
                for idx, lower_signature in enumerate(
                    lower_priority_rules["signatures"][func_name]
                ):
                    if match_input_signature(
                        higher_signature["inputs"], lower_signature["inputs"], "exact"
                    ):
                        # If there's an exact match, update the entry with the higher priority rule
                        lower_priority_rules["signatures"][func_name][
                            idx
                        ] = higher_signature
                        matched = True
                        break
                if not matched:
                    # If there's no match, add the new signature
                    lower_priority_rules["signatures"][func_name].append(
                        higher_signature
                    )

    # For other sections, use dictionary update to overwrite or add new entries
    sections = [
        "templates",
        "commutative_functions_to_convert_to_binomial",
        "functions",
        "transforms",
        "function_logic_dags",
    ]
    for section in sections:
        if section in higher_priority_rules:
            lower_priority_rules[section].update(higher_priority_rules[section])


def get_parent_data_types(G, node_id):
    parents: List[int] = dags.get_ordered_parent_ids(G, node_id)
    parent_data_types = [G.nodes[parent_id]["data_type"] for parent_id in parents]
    return parent_data_types


def match_first_signature__node(G, node_id, converion_rules) -> Optional[Dict]:
    """
    Finds the first matching signature for the given node from within the
    conversion rules dicationary.
    Returns None if no matching signature is found.
    """
    function_name = G.nodes[node_id]["function_name"]
    parent_data_types = get_parent_data_types(G, node_id)
    return match_first_signature(
        converion_rules, function_name, parent_data_types, "permissive"
    )


def match_first_signature(converion_rules, function_name, input_data_types, match_mode):
    matching_signatures = converion_rules["signatures"].get(function_name)
    if not matching_signatures:
        return None
    for signature in matching_signatures:
        if match_input_signature(input_data_types, signature["inputs"], match_mode):
            # if more than one signature matches, the first one will get returned.
            # note this means that the order of signatures within the same function name
            # can matter in conversion_rules files. In general more specific signature definitions
            # should be listed first.
            return signature
    return None


def get_functions_without_conversion_instructions(
    signature_definitions, required_signatures
):
    missing_signatures = {"signatures": {}}

    for func_name, func_required_signatures in required_signatures[
        "signatures"
    ].items():

        for required_signature in func_required_signatures:
            # Check if the current signature matches any in the signature_definitions
            match_found = False
            for signature in signature_definitions["signatures"].get(func_name, []):
                if match_input_signature(
                    required_signature["inputs"], signature["inputs"], "permissive"
                ):
                    if "no_code" not in signature:
                        match_found = True
                        break

            # If no match is found, copy the required_signature
            if not match_found:
                # Check if the function already exists in missing_signatures, if not, initialize it
                if func_name not in missing_signatures["signatures"]:
                    missing_signatures["signatures"][func_name] = []

                missing_signatures["signatures"][func_name].append(required_signature)

    # Validation to check if the modified signature dictionary is valid
    assert validation.is_valid_signature_definition_dict(
        missing_signatures, False, False
    ), "Signature dictionary is not valid."
    return missing_signatures


def build_current_signature_definitions(G):
    """
    This builds a list of signature that are currently used by the graph
    """
    signatures = {"signatures": {}}
    for node_id, attributes in G.nodes(data=True):
        if attributes["node_type"] != "function":
            continue
        # Skip function arrays as they are a special case as a tool to
        # allow some function to return multiple outputs in a graph that otherwise
        # expects a single output per function
        if attributes["function_name"] == "FUNCTION_ARRAY":
            continue
        match_sig = match_first_signature__node(G, node_id, signatures)
        if match_sig:
            if match_sig.get("source") == "Required signatures":
                match_sig["locations"].append(
                    dags.node_location_description(G, node_id)
                )
            continue
        parent_data_types = get_parent_data_types(G, node_id)
        return_type = attributes["data_type"]
        additional_params = {"locations": [dags.node_location_description(G, node_id)]}
        add_function_signature(
            signatures,
            attributes["function_name"],
            parent_data_types,
            [return_type],
            "Required signatures",
            False,
            additional_params,
        )
    assert validation.is_valid_signature_definition_dict(
        signatures, False, False
    ), "Signature dictionary is not valid."
    return signatures


def if_missing_save_sigs_and_err(signature_definitions, G):
    """
    checks for signatures missing instructions for how to convert them
    if found, saves them to ./errors/missing_signatures.json and raises error.
    otherwise returns False (good, i.e. nothing missing)
    """
    current_required_signatures = build_current_signature_definitions(G)
    missing_signatures = get_functions_without_conversion_instructions(
        signature_definitions, current_required_signatures
    )
    if len(missing_signatures["signatures"]) > 0:
        with open("./errors/missing_signatures.json", "w") as f:
            json.dump(missing_signatures, f, indent=2)
            raise ValueError(
                "Missing conversion instructions for some functions. Missing signatures saved to ./errors/missing_signatures.json"
            )


def add_function_signature(
    signature_definition_dict,
    function_name,
    input_data_types,
    return_data_types,
    source,
    no_code,
    additional_params={},
):
    """
    Adds a new function signature to the specified dictionary of function signatures.
    Supports the inclusion of additional metadata and flags, such as the source of the
    signature. The no_code input will add the "no_code" flag to the signature. The purpose
    of this flag is to indicate that the signature intentionally has no code associated with it
    which is important for validating the data structure.
    """
    if not function_name in signature_definition_dict["signatures"]:
        signature_definition_dict["signatures"][function_name] = []
    signature_definition_dict["signatures"][function_name].append(
        {"inputs": input_data_types, "outputs": return_data_types, "source": source}
    )
    if no_code:
        signature_definition_dict["signatures"][function_name][-1]["no_code"] = True
    for k, v in additional_params.items():
        signature_definition_dict["signatures"][function_name][-1][k] = v


def match_type(type1, type2, is_ordered, is_strict):
    # Define a helper function to apply checks, so order doesn't matter for certain conditions.
    def is_match(t1, t2):
        # Any means any data type that represents a single value--not an array or table column
        any_data_types = ["Text", "Number", "Boolean", "Date"]
        # return true if either side is None is used when matching transform patterns.
        return t1 is None or (t1 in any_data_types and t2 == "Any") or t1 == t2

    if is_strict:
        return type1 == type2

    # Check for Range and ARRAY/TABLE_COLUMN match regardless of the order.
    if "Range" in [type1, type2] and (
        type1.startswith(("ARRAY", "TABLE_COLUMN"))
        or type2.startswith(("ARRAY", "TABLE_COLUMN"))
    ):
        return True

    if is_ordered:
        return is_match(type1, type2)

    return is_match(type1, type2) or is_match(type2, type1)


def match_input_signature(parent_data_types, input_signature, match_mode):
    """
    Determines if the provided parent data types match the input signature based on the
    specified match mode. An exact match is jsut parent_data_types == input_signature;
    however use of this functions is preferred over that.
    Note that except in the case of an exact match, parent_data_types and input_signature
    are NOT interchangeable. Specifically, input_signature can have "Mutliple" to match to
    multiple parameters in parent data types.

    Parameters:
    - parent_data_types (list of str): The data types of the parent inputs to be matched.
    - input_signature (list of str): The expected data types as defined in the function's input signature.
    - match_mode (str): The level of strictness for the match, which can be one of the following:
        - "exact": Requires the parent data types and input signature to be exactly the same.
        - "strict": Allows for flexibility of using "Multiple" in the input signature
        - "permissive": Allows further flexibility to count a match if either side is blank or "Any". Also matches ranges, arrays and table columns as the same

    Returns:
    - bool: True if the parent data types match the input signature according to the specified match type,
            False otherwise.


    Note:
    - The function internally handles special cases like "Multiple[..]" in the input signature, allowing for a variable
      number of inputs of a specified type.
    """
    valid_match_mode_types = "exact", "strict", "permissive"
    if not match_mode in valid_match_mode_types:
        raise ValueError(
            "match_mode must be one of the following: "
            + ", ".join(valid_match_mode_types)
        )

    if match_mode == "exact":
        return parent_data_types == input_signature

    is_strict = match_mode == "strict"

    remaining_data_type = None
    i = 0
    j = 0
    while i < len(parent_data_types) and j < len(input_signature):
        input_type = input_signature[j]
        parent_type = parent_data_types[i]
        if match_type(parent_type, input_type, True, is_strict):
            i += 1
            j += 1
            continue
        # once input_signature has given us a remaining data type, we can't have any more data types from the input signature.
        # it has to be last.
        if input_type.startswith("Multiple["):
            remaining_data_type = input_type[9:-1]
        if remaining_data_type:
            if not match_type(parent_type, remaining_data_type, True, is_strict):
                return False
            else:
                i += 1
        else:
            return False
    # they both need to complete the above together.
    if not (
        (
            i == len(parent_data_types)
            and j == len(input_signature)
            and not remaining_data_type
        )
        or (
            i == len(parent_data_types)
            and j + 1 == len(input_signature)
            and remaining_data_type
        )
    ):
        return False

    return True


def add_signatures_to_library(
    new_sig_dict, lib_sig_dict, source, allow_multiple_outputs
):
    """
    Note the plural: signatures. This adds a one set of signatures to a 'library'
    of signatures. lib_sig_digt is the base and new_sig_dict is what is getting added
    along with the name attached in "source"
    """
    assert validation.is_valid_signature_definition_dict(
        new_sig_dict, allow_multiple_outputs, False
    ), "invalid signature dictionary"
    assert validation.is_valid_signature_definition_dict(
        lib_sig_dict, allow_multiple_outputs, False
    ), "invalid signature dictionary"

    new_sigs = new_sig_dict["signatures"]
    signature_definition_library = lib_sig_dict["signatures"]

    for key, value in new_sigs.items():
        if key not in signature_definition_library:
            signature_definition_library[key] = []

        for item in value:
            found = False
            for existing_item in signature_definition_library[key]:
                if (
                    match_input_signature(
                        item["inputs"], existing_item["inputs"], "exact"
                    )
                    and item["outputs"]
                    == existing_item[
                        "outputs"
                    ]  # a library can have different outptus for the same input signature.
                ):
                    if source is not None:
                        if "source" not in existing_item:
                            existing_item["source"] = [source]
                        else:
                            existing_item["source"].append(source)

                    found = True
                    break

            if not found:
                item["source"] = [source]
                signature_definition_library[key].append(item)


def _match_all_signatures(
    conversion_rules, function_name, input_data_types, can_have_multiple_outputs
):
    """
    used by get_data_types today.
    Matches function signatures against provided parameters and returns all matches.

    :param conversion_rules: Dictionary of function signaturesa, may also include the info needed for translating them.
    :param function_name: Name of the function to match signatures for.
    :param input_data_types: signature inputs to match on.
    :return: A list of tuples. Each tuple has (1) the return type function signature and (2) the list of sources for that signature.
             Returns an empty list if no match is found.
    """
    matches = []
    signatures = conversion_rules["signatures"].get(function_name)

    if signatures is not None:
        for signature in signatures:
            if match_input_signature(
                input_data_types, signature["inputs"], "permissive"
            ):
                if len(signature["outputs"]) > 1 and not can_have_multiple_outputs:
                    continue
                outputs = signature["outputs"]
                source = signature.get("source", None)
                matches.append((outputs, signature["inputs"], source))

    return matches


def get_data_types(
    G,
    node_id,
    conversion_rules,
    signature_definition_library,
    auto_add,
    conversion_tracker,
) -> List[str]:
    """
    This is an important step for graph. We need to set the data type for each node where it isn't present.
    This function defines the data_types for the given node based on the function name and parent data types
    using the conversion_rules and signature definition library. If missing from the conversion_rules,
    the signature definition library can be used to add it to conversion rules. If not match is found
    the user is asked to supply
    """

    def generate_options_message(matching_tuples):
        message_parts = []
        valid_responses = []
        for index, (return_types, sig_inputs, source) in enumerate(
            matching_tuples, start=1
        ):
            message_parts.append(
                f"{index}) {', '.join(source)} with input type {', '.join(sig_inputs)} and return type(s) {return_types}"
            )
            valid_responses.append(str(index))
        message_parts.append(f"{len(matching_tuples) + 1}) None of these, don't add")
        valid_responses.append(str(len(matching_tuples) + 1))

        message = "\n".join(message_parts)
        return message, valid_responses

    
    assert validation.is_valid_conversion_rules_dict(
        conversion_rules
    ), "signature is not valid"
    assert validation.is_valid_signature_definition_dict(
        signature_definition_library, True, True
    ), "signature dictionary is not valid"

    function_name = G.nodes[node_id]["function_name"]

    if function_name == "FUNCTION_ARRAY":
        multiple_output_node_id, position = (
            get_parent_function_and_position_for_function_array_node(G, node_id)
        )
        multiple_output_data_types = G.nodes[multiple_output_node_id]["data_types"]
        return [multiple_output_data_types[position - 1]]

    parent_data_types = get_parent_data_types(G, node_id)

    can_have_multiple_outputs = validation.node_can_have_multiple_outputs(G, node_id)
    matching_tuples = _match_all_signatures(
        conversion_rules, function_name, parent_data_types, can_have_multiple_outputs
    )
    if len(matching_tuples) > 1:
        # conversion file should have signatures ordered so that when there are multiple matches we use the first one (which typically is the most specific one)
        matching_tuples = matching_tuples[:1]

    if len(matching_tuples) == 1:
        return_types, sig_inputs, _ = matching_tuples[0]
        ct.update_conversion_tracker_sig(
            conversion_tracker,
            function_name,
            sig_inputs,
            return_types,
            "return_types_assigned",
        )
        return return_types

    missing_signature_info = f"function name: {function_name} with input signature {', '.join(parent_data_types)}"

    matching_tuples = _match_all_signatures(
        signature_definition_library,
        function_name,
        parent_data_types,
        can_have_multiple_outputs,
    )
    if matching_tuples:
        if auto_add and len(matching_tuples) == 1:
            return_types, sig_inputs, sources = matching_tuples[0]
            add_function_signature(
                conversion_rules,
                function_name,
                sig_inputs,
                return_types,
                sources,
                True,
            )
            ct.update_conversion_tracker_sig(
                conversion_tracker,
                function_name,
                sig_inputs,
                return_types,
                "func_auto_added_and_return_types_assigned",
            )
            return return_types
        if len(matching_tuples) == 1:
            return_types, sig_inputs, sources = matching_tuples[0]
            resp = ui.ask_question(
                f"Signature for {missing_signature_info} not found in current signature dictionary. Match found in the provided library: {', '.join(sources)} with return type(s) {return_types}. Do you want to add it?",
                ["y", "n"],
            )
            if resp == "y":
                add_function_signature(
                    conversion_rules,
                    function_name,
                    sig_inputs,
                    return_types,
                    sources,
                    True,
                )
                ct.update_conversion_tracker_sig(
                    conversion_tracker,
                    function_name,
                    sig_inputs,
                    return_types,
                    "manual_add_from_library_and_return_types_assigned",
                )
                return return_types
        else:
            options_message, valid_responses = generate_options_message(matching_tuples)
            resp = ui.ask_question(
                f"Signature for {missing_signature_info} not found in current signature dictionary. Match found in {len(matching_tuples)} provided libraries:\n{options_message}",
                valid_responses,
            )
            if resp != str(
                len(matching_tuples) + 1
            ):  # if not the option to add manually
                selected_tuple = matching_tuples[int(resp) - 1]
                return_types, sig_inputs, sources = selected_tuple
                add_function_signature(
                    conversion_rules,
                    function_name,
                    sig_inputs,
                    return_types,
                    sources,
                    True,
                )
                ct.update_conversion_tracker_sig(
                    conversion_tracker,
                    function_name,
                    sig_inputs,
                    return_types,
                    "manual_add_from_library_and_return_types_assigned",
                )
                return return_types

    resp = ui.ask_question(
        f"Signature for {missing_signature_info} not found. Dag: {G.graph['name']}. Do you want to add it manually (m) or abort (a)?",
        ["m", "a"],
    )
    if resp == "m":
        valid_data_type_part = partial(validation.valid_data_type, is_strict=False)
        return_type = ui.ask_question_validation_function(
            f"Signature for {missing_signature_info}, what should the return data type be. Note: multiple outputs not supported for manual adds? Examples: Text, Number, Boolean, Date, ARRAY[Text], TABLE_COLUMN[Number]",
            valid_data_type_part,
        )
        if return_type is not None:
            add_function_signature(
                conversion_rules,
                function_name,
                parent_data_types,
                [return_type],
                "manual",
                True,
            )
            ct.update_conversion_tracker_sig(
                conversion_tracker,
                function_name,
                parent_data_types,
                [return_type],
                "return_type_manually_added_and_return_type_assigned",
            )
            return [return_type]

    errs.save_dag_and_conversion_rules_and_raise__node(
        G,
        node_id,
        conversion_rules,
        f"Signature for {missing_signature_info} not found. Node {node_id} in tree: {G.graph['name']}. Aborting",
    )
    return ["won't get here"]


def sort_signatures(conversion_rules):
    sorted_dict = {
        k: conversion_rules["signatures"][k]
        for k in sorted(conversion_rules["signatures"].keys())
    }
    conversion_rules["signatures"] = sorted_dict


def get_parent_function_and_position_for_function_array_node(G, node_id):
    parents = dags.get_ordered_parent_ids(G, node_id)
    multiple_output_function_node_id = parents[0]
    position_node_id = parents[1]
    position = int(G.nodes[position_node_id]["value"])
    return multiple_output_function_node_id, position


def create_signature_dictionary(function_logic_dags):
    new_sig_dict = initialize_conversion_rules()
    new_sigs = new_sig_dict["signatures"]

    for func_name, dag in function_logic_dags.items():
        # Extract input and output node IDs
        input_node_ids = dag.graph["input_node_ids"]
        output_node_ids = dag.graph["output_node_ids"]

        # Extract data types for inputs and outputs
        inputs = [dag.nodes[node_id]["data_type"] for node_id in input_node_ids]
        outputs = [dag.nodes[node_id]["data_type"] for node_id in output_node_ids]

        new_signature = {"inputs": inputs, "outputs": outputs}
        new_sigs[func_name] = [new_signature]

    return new_sig_dict


def load_and_deserialize_rules(file_name: str) -> Dict[str, Any]:
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"{file_name} not found.")

    with open(file_name, "r") as file:
        loaded_dict = json.load(file)

    deserialized_rules = deserialize_dict_with_dags(loaded_dict)
    if not validation.is_valid_conversion_rules_dict(deserialized_rules):
        raise ValueError(f"Conversion rules in {file_name} are not valid.")

    return deserialized_rules


def serialize_and_save_rules(
    conversion_rules: Dict[str, Any], conversion_rules_file: str
) -> bool:
    sort_signatures(conversion_rules)
    serialized_conversion_rules = serialize_dict_with_dags(conversion_rules)
    with open(conversion_rules_file, "w") as f:
        json.dump(serialized_conversion_rules, f, indent=2)
        return True


def serialize_dict_with_dags(data):
    """
    Serialize a dictionary that may contain NetworkX MultiDiGraphs to a JSON-compatible format.
    """

    def serialize_helper(item):
        if isinstance(item, nx.MultiDiGraph):
            return {"_is_graph": True, "graph_data": nx.node_link_data(item)}
        elif isinstance(item, dict):
            return {key: serialize_helper(value) for key, value in item.items()}
        elif isinstance(item, list):
            return [serialize_helper(element) for element in item]
        else:
            return item

    return serialize_helper(data)


def deserialize_dict_with_dags(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize a JSON-compatible structure into a dictionary, converting serialized graph structures back to NetworkX MultiDiGraphs.
    """

    def deserialize_helper(item):
        if isinstance(item, dict):
            if item.get("_is_graph"):
                return nx.node_link_graph(item["graph_data"])
            else:
                return {key: deserialize_helper(value) for key, value in item.items()}
        elif isinstance(item, list):
            return [deserialize_helper(element) for element in item]
        else:
            return item

    result = deserialize_helper(data)
    if not isinstance(result, dict):
        raise ValueError("Deserialized data is not a dictionary as expected.")
    return result


def filter_conversion_rules_by_conv_tracker(conversion_rules, conversion_tracker):
    assert validation.is_valid_conversion_tracker(conversion_tracker)
    assert validation.is_valid_conversion_rules_dict(conversion_rules)

    filtered_conversion_rules = initialize_conversion_rules()

    # 1. Add conversion_rules signatures that were used as recorded in the conversion tracker
    used_signatures = conversion_tracker["signatures"]
    sig_to_build = filtered_conversion_rules["signatures"]
    sig_library = conversion_rules["signatures"]

    for func_name, sig_list in used_signatures.items():
        for sig in sig_list:
            match_found = False
            lib_sigs = sig_library[func_name]
            for lib_sig in lib_sigs:
                if match_input_signature(lib_sig["inputs"], sig["inputs"], "exact"):
                    if not func_name in sig_to_build:
                        sig_to_build[func_name] = []
                    sig_to_build[func_name].append(lib_sig)
                    match_found = True
                    break
            if not match_found:
                raise ValueError(
                    f"Signature {sig} is conversion_tracker but not found in library for function {func_name}. Doesn't make sense"
                )

    # 2. add templates that were used
    used_templates = conversion_tracker["templates_used"]
    templates_to_build = filtered_conversion_rules["templates"]
    template_library = conversion_rules.get("templates", {})

    for template_name, _ in used_templates.items():
        templates_to_build[template_name] = template_library[template_name]

    # 3. add binomial expansions
    used_binomial_expansions = conversion_tracker["binomial_expansions"]
    binomial_expansions_to_build = filtered_conversion_rules[
        "commutative_functions_to_convert_to_binomial"
    ]
    binomial_expansions_library = conversion_rules.get(
        "commutative_functions_to_convert_to_binomial", {}
    )

    for func_name, _ in used_binomial_expansions.items():
        binomial_expansions_to_build[func_name] = binomial_expansions_library[func_name]

    # 4. add functions
    used_functions = conversion_tracker["used_functions"]
    functions_to_build = filtered_conversion_rules["functions"]
    functions_library = conversion_rules.get("functions", {})

    for func_name, _ in used_functions.items():
        functions_to_build[func_name] = functions_library[func_name]

    # 5. add transforms
    used_transforms = conversion_tracker["transforms"]
    transforms_to_build = filtered_conversion_rules["transforms"]
    transforms_library = conversion_rules["transforms"]

    for transform_name, _ in used_transforms.items():
        transform_name = (
            transform_name.upper()
        )  # transforms in conversion_rules are always upper case, but conversion_tracker uses whatever case there is from the name property.
        transforms_to_build[transform_name] = transforms_library[transform_name]

    # 6. add function_logic_dags
    used_function_logic_dags = conversion_tracker["expanded_functions"]
    function_logic_dags_to_build = filtered_conversion_rules["function_logic_dags"]
    function_logic_dags_library = conversion_rules["function_logic_dags"]

    for func_name, _ in used_function_logic_dags.items():
        function_logic_dags_to_build[func_name] = function_logic_dags_library[func_name]

    return filtered_conversion_rules


def main():
    # Test case as described
    b = ["Multiple[Number]"]
    a = ["Number", "Number", "Number"]

    # Run the test case
    result = match_input_signature(a, b, "strict")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
