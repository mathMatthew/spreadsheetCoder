
from typing import Any, Tuple, Dict, List

import validation

def empty_conversion_tracker() -> Dict[str, Any]:
    return {
        "func_sigs": {},
        "events": {},
        "transforms": {},
        "expanded_functions": {},
        "binomial_expansions": {},
        "templates_used": {},
    } 

def update_conversion_tracker_record_binomial_expansion(
    conversion_tracker, function_name, binomial_record
):
    # right now all I am recording is the count. inluding
    # binomial record in case we want to have this in the future.
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Invalid conversion tracker"
    conversion_tracker["binomial_expansions"][function_name] = (
        conversion_tracker["binomial_expansions"].get(function_name, 0) + 1
    )

def update_conversion_tracker_template_used(
    conversion_tracker, template_name
):
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Invalid conversion tracker"
    conversion_tracker["templates_used"][template_name] = (
        conversion_tracker["templates_used"].get(template_name, 0) + 1
    )


def update_conversion_tracker_record_transform(conversion_tracker, transform_name):
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Invalid conversion tracker"
    conversion_tracker["transforms"][transform_name] = (
        conversion_tracker["transforms"].get(transform_name, 0) + 1
    )


def update_conversion_tracker_record_expand(conversion_tracker, expanded_function_name):
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Invalid conversion tracker"
    conversion_tracker["expanded_functions"][expanded_function_name] = (
        conversion_tracker["expanded_functions"].get(expanded_function_name, 0) + 1
    )


def update_conversion_tracker_sig(
    conversion_tracker, function_name, parent_data_types, return_types, event_name
):
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Invalid conversion tracker"

    if function_name not in conversion_tracker["func_sigs"]:
        # If function name isn't there, add it with record of usage.
        conversion_tracker["func_sigs"][function_name] = [
            {"inputs": parent_data_types, "outputs": return_types, event_name: 1}
        ]
    else:
        for sig in conversion_tracker["func_sigs"][function_name]:
            if sig["inputs"] == parent_data_types and sig["outputs"] == return_types:
                sig[event_name] = sig.get(event_name, 0) + 1
                return

        # If function name is there, but no matching signature was found, add the new signature with usage.
        new_signature = {
            "inputs": parent_data_types,
            "outputs": return_types,
            event_name: 1,
        }
        conversion_tracker["func_sigs"][function_name].append(new_signature)

def update_conversion_tracker_event(conversion_tracker, event_name):
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    if event_name not in conversion_tracker["events"]:
        conversion_tracker["events"][event_name] = {"usage_count": 1}
    else:
        event = conversion_tracker["events"][event_name]
        event["usage_count"] = event.get("usage_count", 0) + 1
