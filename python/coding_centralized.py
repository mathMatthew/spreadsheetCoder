from typing import Any, Dict, Tuple, List, Callable, Set, Optional
import re

import conv_tracker as ct
import errs, validation, dags
import signatures as sigs

def code_cached_node(
    G, node_id, conversion_tracker, conversion_func_sigs, replace_key_fn
):
    if G.nodes[node_id]["node_type"] == "input":
        if "output_name" in G.nodes[node_id]:
            if G.nodes[node_id]["output_name"] == G.nodes[node_id]["input_name"]:
                return ""  # maybe we should have a required template for this situation and langauges where they need nothing, define the required tmeplate as an empty string.will implement if needed.
            else:
                template = conversion_func_sigs["templates"]["cache_default"][
                    "template"
                ]  # maybe this should be an optional template and if we don't have it then use the default. will implement if needed.
                code = replace_placeholders(template, replace_key_fn)
                return code
        else:
            return ""  # thinking is that there is nothing additional to do here. no special need to 'cache' them. AT least that is true for the SQL. May have need for other languages.

    function_name = G.nodes[node_id]["function_name"]

    parent_data_types = sigs.get_parent_data_types(G, node_id)

    function_signature = sigs.match_signature(G, node_id, conversion_func_sigs)
    if not function_signature:
        errs.save_dag_and_raise_node(
            G,
            node_id,
            f"Unsupported function: {function_name} with input data types {', '.join(parent_data_types)} at node id: {node_id}",
        )
        return ""

    if len(function_signature["outputs"]) > 1:
        raise ValueError(
            f"Add support for more than one output. Function {function_name} requires >1 output"
        )
        return ""

    ct.update_conversion_tracker_sig(
        conversion_tracker,
        function_name,
        function_signature["inputs"],
        function_signature["outputs"],
        "code_function_cached",
    )

    template_key = function_signature.get("template", "cache_default")
    template = conversion_func_sigs["templates"][template_key]["template"]
    code = replace_placeholders(template, replace_key_fn)
    ct.update_conversion_tracker_template_used(conversion_tracker, template_key)
    return code

def code_std_function_node(
    G,
    node_id,
    conversion_tracker,
    supported_functions,
    code_node_fn,
    replace_key_fn,
    special_process_fn,
) -> str:
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    signature = sigs.match_signature(G, node_id, supported_functions)
    if not signature:
        function_name = G.nodes[node_id]["function_name"]
        parent_data_types = sigs.get_parent_data_types(G, node_id)
        errs.save_dag_and_raise_node(
            G,
            node_id,
            f"Unsupported function: {function_name} with input data types {', '.join(parent_data_types)} at node id: {node_id}",
        )
        return ""

    return code_supported_function(
        G,
        node_id,
        signature,
        supported_functions,
        conversion_tracker,
        code_node_fn,
        replace_key_fn,
        special_process_fn,
    )

def code_supported_function(
    G,
    node_id,
    function_signature,
    supported_functions,
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
        #should really never happen now that we created the missing_signature stuff.
        if "source" in function_signature:
            source = function_signature["source"]
            source = " and ".join(source) if isinstance(source, list) else source
            if source == "manual":
                msg = f'The signature for {function_name} with inputs {", ".join(function_signature["inputs"])} was added manually. Right now there is no mechanism to code it. Add {function_name} with inputs {", ".join(function_signature["inputs"])} to the supported_functions json. Node id: {node_id}'
            else:
                msg = f'Signature from { source } exists for {function_name} but the signature has no mechanism to code it. Add {function_name} with inputs {", ".join(function_signature["inputs"])} to the supported_functions json. Node id: {node_id}'
        else:
            msg = f'Signature exists for {function_name} but the signature has no mechanism to code it. Add {function_name} with inputs {", ".join(function_signature["inputs"])} to the supported_functions json. Node id: {node_id}'

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
        template = supported_functions["templates"][template_name]["template"]
        code = replace_placeholders(template, replace_key_fn)
        ct.update_conversion_tracker_template_used(
            conversion_tracker, template_name
        )
    else:
        code = function_signature.get("code_before", "")
        if "operator" in function_signature:
            if len(parents) > 2:
                errs.save_dag_and_raise_node(
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

    special_process_fn(G, node_id, function_signature, conversion_tracker)

    return code

def replace_placeholders(
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

