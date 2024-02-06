import json
from typing import Any, Dict, Tuple, List, Optional

import validation, dags, ui, errs
import conv_tracker as ct


def empty_func_sigs() -> Dict[str, Any]:
    return {
        "signatures": {},
    }


def get_parent_data_types(G, node_id):
    parents: List[int] = dags.get_ordered_parent_ids(G, node_id)
    parent_data_types = [G.nodes[parent_id]["data_type"] for parent_id in parents]
    return parent_data_types


def match_signature(G, node_id, supported_functions) -> Optional[Dict]:
    function_name = G.nodes[node_id]["function_name"]
    matching_signatures = supported_functions["signatures"].get(function_name)
    parent_data_types = get_parent_data_types(G, node_id)
    if not matching_signatures:
        return None
    for signature in matching_signatures:
        if match_input_signature(parent_data_types, signature["inputs"], True):
            return signature
    return None


def get_functions_without_conversion_instructions(
    signature_definitions, required_signatures
):
    missing_signatures = {"signatures": {}}

    for func_name, func_required_signatures in required_signatures["signatures"].items():
        if func_name == "INDEX":
            print("checking index")

        for required_signature in func_required_signatures:
            # Check if the current signature matches any in the signature_definitions
            match_found = False
            for signature in signature_definitions["signatures"].get(func_name, []):
                if match_input_signature(required_signature["inputs"], signature["inputs"], True):
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
    assert validation.is_valid_fn_sig_dict(
        missing_signatures, False
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
        match_sig = match_signature(G, node_id, signatures)
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
    assert validation.is_valid_fn_sig_dict(
            signatures, False
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
    func_sigs,
    function_name,
    parent_data_types,
    return_data_types,
    source,
    no_code,
    additional_params={},
):
    if not function_name in func_sigs["signatures"]:
        func_sigs["signatures"][function_name] = []
    func_sigs["signatures"][function_name].append(
        {"inputs": parent_data_types, "outputs": return_data_types, "source": source}
    )
    if no_code:
        func_sigs["signatures"][function_name][-1]["no_code"] = True
    for k, v in additional_params.items():
        func_sigs["signatures"][function_name][-1][k] = v


def match_type(type1, type2, strict):
    # Define a helper function to apply checks, so order doesn't matter.
    def is_match(t1, t2):
        # Any means any data type that represents a single value--not an array or table column
        any_data_types = ["Text", "Number", "Boolean", "Date"]
        return (
            t1 is None
            or (t1 == "Any" and t2 in any_data_types)
            or (
                t1 == "Range"
                and (t2.startswith("ARRAY") or t2.startswith("TABLE_COLUMN"))
            )
        )

    if not strict:
        # Apply the checks for both (type1, type2) and (type2, type1)
        if is_match(type1, type2) or is_match(type2, type1):
            return True

    # Strict comparison
    return type1 == type2


def match_input_signature(parent_data_types, input_signature, strict):
    remaining_data_type = None
    i = 0
    j = 0
    while i < len(parent_data_types) and j < len(input_signature):
        input_type = input_signature[j]
        # once input_signature has given us a remaining data type, we can't have any more data types from the input signature.
        if input_type.startswith("Multiple["):
            remaining_data_type = input_type[9:-1]
        parent_type = parent_data_types[i]
        if remaining_data_type:
            if not match_type(parent_type, remaining_data_type, strict):
                return False
            else:
                i = i + 1
        else:
            if not match_type(parent_type, input_type, strict):
                return False
            else:
                i = i + 1
                j = j + 1
    #they both need to complete the above together.
    if not(
        (i == len(parent_data_types) and j == len(input_signature) and not remaining_data_type)
        or (i == len(parent_data_types) and j+1 == len(input_signature) and remaining_data_type)
    ):
        return False

    return True


def add_signatures_to_library(new_sig_dict, lib_sig_dict, source):
    """
    think of the library as the base and new_sig_dict as what is getting added
    along with the name attached in "source"
    """
    assert validation.is_valid_fn_sig_dict(new_sig_dict, True), "invalid signature dictionary"
    assert validation.is_valid_fn_sig_dict(lib_sig_dict, False), "invalid signature dictionary"

    new_sigs = new_sig_dict["signatures"]
    library_func_sigs = lib_sig_dict["signatures"]

    for key, value in new_sigs.items():
        if key not in library_func_sigs:
            library_func_sigs[key] = []

        for item in value:
            found = False
            for existing_item in library_func_sigs[key]:
                if (
                    item["inputs"] == existing_item["inputs"]
                    and item["outputs"] == existing_item["outputs"]
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
                library_func_sigs[key].append(item)


def match_function_signature(fn_sig_translation_dict, function_name, parent_data_types, can_have_multiple_outputs):
    """
    Matches function signatures against provided parameters and returns all matches.

    :param fn_sig_translation_dict: Dictionary of function signaturesa, may also include the info needed for translating them.
    :param function_name: Name of the function to match signatures for.
    :param parent_data_types: signature inputs to match on.
    :return: A list of tuples. Each tuple has (1) the return type function signature and (2) the list of sources for that signature.
             Returns an empty list if no match is found.
    """
    matches = []
    signatures = fn_sig_translation_dict["signatures"].get(function_name)

    if signatures is not None:
        for signature in signatures:
            if match_input_signature(parent_data_types, signature["inputs"], False):
                if len(signature["outputs"]) > 1 and not can_have_multiple_outputs:
                    continue
                outputs = signature["outputs"]
                source = signature.get("source", None)
                matches.append((outputs, signature["inputs"], source))

    return matches



def get_data_types(
    G, node_id, conversion_signatures, library_func_sigs, auto_add, conversion_tracker
) -> List[str]:
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

    assert validation.is_valid_conversion_fn_sig_dict(
        conversion_signatures
    ), "signature is not valid"
    assert validation.is_valid_fn_sig_dict(library_func_sigs, True), "signature dictionary is not valid"

    function_name = G.nodes[node_id]["function_name"]
    if function_name == "FUNCTION_ARRAY":
        multiple_output_node_id, position = get_parent_function_and_position_for_function_array_node(G, node_id)
        multiple_output_data_types = G.nodes[multiple_output_node_id]["data_types"]
        return [multiple_output_data_types[position-1]]

    parent_data_types = get_parent_data_types(G, node_id)

    can_have_multiple_outputs = validation.node_can_have_multiple_outputs(G, node_id)
    matching_tuples = match_function_signature(
        conversion_signatures, function_name, parent_data_types, can_have_multiple_outputs
    )
    if len(matching_tuples) > 1:
        msg = f"the conversion signature file has multiple matches for {function_name} with input signature {', '.join(parent_data_types)}"
        errs.save_function_signatures_and_raise(conversion_signatures, msg)

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

    matching_tuples = match_function_signature(
        library_func_sigs, function_name, parent_data_types, can_have_multiple_outputs
    )
    if matching_tuples:
        if auto_add and len(matching_tuples) == 1:
            return_types, sig_inputs, sources = matching_tuples[0]
            add_function_signature(
                conversion_signatures,
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
                    conversion_signatures,
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
            if resp != str(len(matching_tuples) + 1): #if not the option to add manually
                selected_tuple = matching_tuples[int(resp) - 1]
                return_types, sig_inputs, sources = selected_tuple
                add_function_signature(
                    conversion_signatures,
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
        return_type = ui.ask_question_validation_function(
            f"Signature for {missing_signature_info}, what should the return data type be. Note: multiple outputs not supported for manual adds? Examples: Text, Number, Boolean, Date, ARRAY[Text], TABLE_COLUMN[Number]",
            validation.valid_data_type_strict,
        )
        if return_type is not None:
            add_function_signature(
                conversion_signatures,
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

    errs.save_dag_and_raise_node(
        G,
        node_id,
        f"Signature for {missing_signature_info} not found. Node {node_id} in tree: {G.graph['name']}. Aborting",
    )
    return ["won't get here"]

def get_parent_function_and_position_for_function_array_node(G, node_id):
    parents = dags.get_ordered_parent_ids(G, node_id)
    multiple_output_function_node_id = parents[0]
    position_node_id = parents[1]
    position = int(G.nodes[position_node_id]["value"])
    return multiple_output_function_node_id, position


def create_signature_dictionary(function_logic_dags):
    new_sig_dict = empty_func_sigs()
    new_sigs = new_sig_dict["signatures"]

    for func_name, dag in function_logic_dags.items():
        # Initialize the signature list for the function if it doesn't exist

        # Extract input and output node IDs
        input_node_ids = dag.graph["input_node_ids"]
        output_node_ids = dag.graph["output_node_ids"]

        # Extract data types for inputs and outputs
        inputs = [dag.nodes[node_id]["data_type"] for node_id in input_node_ids]
        outputs = [dag.nodes[node_id]["data_type"] for node_id in output_node_ids]

        new_signature = {"inputs": inputs, "outputs": outputs}
        new_sigs[func_name] = [new_signature]

    return new_sig_dict
