from typing import Any, Tuple, Dict, List
import conversion_rules as cr
import validation


def initialize_conversion_tracker() -> Dict[str, Any]:
    return {
        "signatures": {},
        "events": {},
        "transforms": {},
        "expanded_functions": {},
        "binomial_expansions": {},
        "templates_used": {},
        "used_functions": {},
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


def update_conversion_tracker_template_used(conversion_tracker, template_name):
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
    conversion_tracker, function_name, input_data_types, return_types, event_name
):
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Invalid conversion tracker"

    if function_name not in conversion_tracker["signatures"]:
        # If function name isn't there, add it with record of usage.
        conversion_tracker["signatures"][function_name] = [
            {"inputs": input_data_types, "outputs": return_types, event_name: 1}
        ]
    else:
        for sig in conversion_tracker["signatures"][function_name]:
            if cr.match_input_signature(input_data_types, sig["inputs"], "strict"):
                sig[event_name] = sig.get(event_name, 0) + 1
                return

        # If function name is there, but no matching signature was found, add the new signature with usage.
        new_signature = {
            "inputs": input_data_types,
            "outputs": return_types,
            event_name: 1,
        }
        conversion_tracker["signatures"][function_name].append(new_signature)


def update_conversion_tracker_event(conversion_tracker, event_name):
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    if event_name not in conversion_tracker["events"]:
        conversion_tracker["events"][event_name] = {"usage_count": 1}
    else:
        event = conversion_tracker["events"][event_name]
        event["usage_count"] = event.get("usage_count", 0) + 1


def update_conversion_tracker_functions(conversion_tracker, add_functions: List[str]):
    assert validation.is_valid_conversion_tracker(conversion_tracker)
    for add_function in add_functions:
        if add_function not in conversion_tracker["used_functions"]:
            conversion_tracker["used_functions"][add_function] = {"usage_count": 1}
        else:
            conversion_tracker["used_functions"][add_function]["usage_count"] += 1
