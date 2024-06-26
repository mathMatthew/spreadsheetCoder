from typing import Any, Dict, Tuple, List, Union, Callable, Set, Optional
import re
from datetime import datetime

import conv_tracker as ct
import errs, validation, dags
import conversion_rules as cr



def convert_to_python_type(value, data_type):
    # move this to a centralized testing module at some point.
    if data_type == "Text":
        return value
    elif data_type == "Number":
        return float(value)
    elif data_type == "Boolean":
        return value.lower() == "true"  # or bool(value)?
    elif data_type == "Date":
        try:
            return datetime.strptime(value, "%m/%d/%Y")
        except ValueError:
            return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p")

    # Add other data types as needed
    return value


def code_persistent_node(
    G, node_id, conversion_tracker, conversion_rules, replace_key_fn
):
    if G.nodes[node_id]["node_type"] == "input":
        if "output_name" in G.nodes[node_id]:
            if G.nodes[node_id]["output_name"] == G.nodes[node_id]["input_name"]:
                code = ""  # maybe we should have a required template for this situation and langauges where they need nothing, define the required tmeplate as an empty string.will implement if needed.
            else:
                template = conversion_rules["templates"]["persist_default"][
                    "force-persist-template"
                ]  # maybe this should be an optional template and if we don't have it then use the default. will implement if needed.
                code = replace_placeholders(template, replace_key_fn)
        else:
            code = ""  # thinking is that there is nothing additional to do here. no special need to 'persist' them. AT least that is true for the SQL. May have need for other languages.
    else:
        function_name = G.nodes[node_id]["function_name"]

        function_signature = cr.match_first_signature__node(G, node_id, conversion_rules)
        if not function_signature:
            input_data_types = cr.get_parent_data_types(G, node_id)
            errs.save_dag_and_raise__node(
                G,
                node_id,
                f"Unsupported function: {function_name} with input data types {', '.join(input_data_types)} at node id: {node_id}",
            )
            code = ""

        else:
            if len(function_signature["outputs"]) > 1:
                raise ValueError(
                    f"Add support for more than one output. Function {function_name} requires >1 output"
                )

            ct.update_conversion_tracker_sig(
                conversion_tracker,
                function_name,
                function_signature["inputs"],
                function_signature["outputs"],
                "code_persistent_node",
            )

            # set the default template
            template_key = "persist_default"

            # but if there is a different persist template, use that instead.
            if "template" in function_signature:
                if conversion_rules["templates"][function_signature["template"]].get(
                    "force-persist", False
                ):
                    template_key = function_signature["template"]

            template = conversion_rules["templates"][template_key]["force-persist-template"]

            code = replace_placeholders(template, replace_key_fn)
            ct.update_conversion_tracker_template_used(conversion_tracker, template_key)
        
    return code


def code_std_function_node(
    G,
    node_id,
    conversion_tracker,
    converion_rules,
    code_node_fn,
    replace_key_fn,
    special_process_fn,
) -> str:
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    signature = cr.match_first_signature__node(G, node_id, converion_rules)
    if not signature:
        function_name = G.nodes[node_id]["function_name"]
        input_data_types = cr.get_parent_data_types(G, node_id)
        errs.save_dag_and_raise__node(
            G,
            node_id,
            f"Unsupported function: {function_name} with input data types {', '.join(input_data_types)} at node id: {node_id}",
        )
        return ""

    return code_supported_function(
        G,
        node_id,
        signature,
        converion_rules,
        conversion_tracker,
        code_node_fn,
        replace_key_fn,
        special_process_fn,
    )


def code_supported_function(
    G,
    node_id,
    function_signature,
    converion_rules,
    conversion_tracker,
    code_node_fn,
    replace_key_fn,
    special_process_fn,
) -> str:
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    function_name = G.nodes[node_id]["function_name"]

    if len(function_signature["outputs"]) > 1:
        raise ValueError(
            f"Add support for more than one output. Function {function_name} requires >1 output"
        )
        return ""

    if function_signature.get("no_code"):
        # should really never happen now that we created the missing_signature stuff.
        if "source" in function_signature:
            source = function_signature["source"]
            source = " and ".join(source) if isinstance(source, list) else source
            if source == "manual":
                msg = f'The signature for {function_name} with inputs {", ".join(function_signature["inputs"])} was added manually. Right now there is no mechanism to code it. Add {function_name} with inputs {", ".join(function_signature["inputs"])} to the converion_rules json. Node id: {node_id}'
            else:
                msg = f'Signature from { source } exists for {function_name} but the signature has no mechanism to code it. Add {function_name} with inputs {", ".join(function_signature["inputs"])} to the converion_rules json. Node id: {node_id}'
        else:
            msg = f'Signature exists for {function_name} but the signature has no mechanism to code it. Add {function_name} with inputs {", ".join(function_signature["inputs"])} to the converion_rules json. Node id: {node_id}'

        errs.save_dag_and_raise_message(G, msg)

    parents = dags.get_ordered_parent_ids(G, node_id)
    ct.update_conversion_tracker_sig(
        conversion_tracker,
        function_name,
        function_signature["inputs"],
        function_signature["outputs"],
        "code_function_standard",
    )

    if "template" in function_signature:
        template_name = function_signature["template"]
        template = converion_rules["templates"][template_name]["no-persist-template"]
        code = replace_placeholders(template, replace_key_fn)
        ct.update_conversion_tracker_template_used(conversion_tracker, template_name)
    else:
        code = function_signature.get("code_before", "")
        if "operator" in function_signature:
            if len(parents) > 2:
                errs.save_dag_and_raise__node(
                    G, node_id, f"Operator functions allowed 2 inputs only."
                )
                return ""
            code += code_node_fn(node_id=parents[0])
            code += f" {function_signature['operator']} "
            if len(parents) == 2:
                code += code_node_fn(node_id=parents[1])
        else:
            code += ", ".join([code_node_fn(node_id=parent) for parent in parents])  # type: ignore
        code += function_signature.get("code_after", "")

    if "add_functions" in function_signature:
        ct.update_conversion_tracker_functions(
            conversion_tracker, function_signature["add_functions"]
        )

    special_process_fn(G, node_id, function_signature, conversion_tracker)

    return code


def replace_placeholders_str(
    template: str, get_placeholder_value: Callable[[str], str]
) -> str:
    missing_variables: Set[str] = set()

    def repl(match: re.Match) -> str:
        nonlocal missing_variables
        placeholder_code: str = match.group(1)
        value = get_placeholder_value(placeholder_code)
        if value is None:
            missing_variables.add(placeholder_code)
            return match.group(0)
        else:
            return value

    pattern = r"<([^<>]*)>"
    replaced_template: str = re.sub(pattern=pattern, repl=repl, string=template)
    # is this needed? replaced_template = replaced_template.replace("\\n", "\n")  # Replace \\n with \n

    if missing_variables:
        raise ValueError(
            f"Missing placeholder variables: {', '.join(missing_variables)}"
        )
    else:
        return replaced_template

def replace_placeholders(
    template: Union[str, Dict[str, Any], List[Any]], get_placeholder_value: Callable[[str], str]
) -> Union[str, Dict[str, Any], List[Any]]:
    if isinstance(template, str):
        return replace_placeholders_str(template, get_placeholder_value)
    elif isinstance(template, dict):
        return {key: replace_placeholders(value, get_placeholder_value) for key, value in template.items()}
    elif isinstance(template, list):
        return [replace_placeholders(item, get_placeholder_value) for item in template]
    else:
        raise TypeError("Template must be either a string, a dictionary, or a list.")
                    