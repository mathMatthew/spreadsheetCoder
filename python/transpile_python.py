"""
This module takes an sc graph and transpiles it to python code
"""
##################################
# Section 1: Imports, constants, global, setup
##################################
import json, os, keyword, dags
import networkx as nx
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Tuple, List
from functools import partial
import pandas as pd
import numpy as np

# internal imports
import setup, validation, errs
import conv_tracker as ct
import coding_centralized as cc
import dag_tables as g_tables
import signatures as sigs

INDENT = " " * 2

supported_function_lib_file = "./system_data/python_supported_functions.json"
add_on_module_file_name = "system_data.python_add_on_functions"
used_modules = set()
used_tables = set()


def get_standard_settings(base_dag_xml_file, working_directory) -> Dict[str, Any]:
    with open(supported_function_lib_file, "r") as f:
        library_python_sigs = json.load(f)
    assert validation.is_valid_fn_sig_dict(
        library_python_sigs
    ), "Function library is not valid."

    standard_paths = setup.get_standard_paths(
        base_dag_xml_file, working_directory
    )
    #override with SQL specific function_logic_dir & transform_dir
    standard_paths["function_logic_dir"] = "./system_data/python_function_logic/"
    standard_paths["transform_logic_dir"] = "./system_data/python_transform_logic/"

    standard_settings = setup.get_standard_settings(standard_paths)

    standard_settings["use_tables"] = True
    # unless the file system has already defined a conversion function dictionary for this file, use the standard library
    if standard_settings["conversion_func_sigs"] is None:
        standard_settings["conversion_func_sigs"] = library_python_sigs

    
    return standard_settings


##################################
# Section 2: Utilities and python helper functions
##################################


def _add_module_to_used_modules(module):
    global used_modules
    used_modules.add(module)


def _add_table_to_used_tables(table_name):
    global used_tables
    used_tables.add(table_name)


def _var_code(G, node_id):
    return "var_" + str(node_id)


def _python_safe_name(name):
    # Check if the name is a Python keyword
    if keyword.iskeyword(name):
        name = name + "_"

    # Ensure the first character is a letter or underscore
    if not name[0].isalpha() and name[0] != "_":
        name = "_" + name

    # Replace invalid characters with underscores
    safe_name = ""
    for char in name:
        if char.isalnum() or char == "_":
            safe_name += char
        else:
            safe_name += "_"

    return safe_name


def _constant_value_in_code(value, value_type):
    if value_type == "Range":
        raise ValueError(f"Add support for range type")
    if value_type == "Any":
        raise ValueError(f"Constant shouldn't have type any")
    if value_type == "Text":
        return f'"{value}"'
    if value_type == "Number":
        return value
    if value_type == "Boolean":
        return value
    if value_type == "Date":
        raise ValueError(f"Add support for date type")
        # consider validating date time stuff in the transform module.

    raise ValueError(f"Unsupported value type: {value_type}")


def _add_indents(text, number):
    # Split the input text into lines
    lines = text.split("\n")

    # Add indent spaces to the beginning of each non-blank line
    indented_lines = [
        INDENT * number + line if line.strip() else line for line in lines
    ]

    # Join the lines back together with newline characters
    indented_text = "\n".join(indented_lines)

    return indented_text


def _convert_to_type(value, data_type):
    # Add conversion logic based on the data_type
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


def _make_header(G, use_tables) -> str:
    header = ""
    for module in used_modules:
        if module == "add_on":
            header += f"import {add_on_module_file_name} as add_on\n"
        else:
            header += f"from {module} import *\n"
    if use_tables:
        header += f"import pandas as pd\n"
    header += "\n"

    tables_dir = G.graph["tables_dir"]
    for import_table in used_tables:
        import_table_file_name = os.path.join(
            tables_dir, f"{_python_safe_name(import_table)}.parquet"
        )
        header += f"df_{import_table} = pd.read_parquet(r'{import_table_file_name}')\n"

    header += "\n"
    header += f'def {_python_safe_name(G.graph["name"])}('
    # create input names
    input_ids: list[Any] = G.graph[
        "input_node_ids"
    ]  # already sorted. see is_valid_graph.
    input_names = [
        _python_safe_name(G.nodes[node_id]["input_name"]) for node_id in input_ids
    ]
    # Joining the names with a comma
    input_names_str = ", ".join(input_names)
    header += f"{input_names_str}):\n"
    return header


##################################
# Section 3: Node processing functions
##################################
def _code_node(G, node_id, is_primary, conversion_func_sigs, conversion_tracker) -> str:
    """
    first entry point to generates code for the node.
    """
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    attribs = G.nodes[node_id]
    node_type = attribs["node_type"]
    data_type = attribs["data_type"]
    if node_type == "input":
        return _python_safe_name(attribs["input_name"])
    if node_type == "constant":
        return _constant_value_in_code(attribs["value"], data_type)
    if node_type == "function":
        if attribs["cache"] and not is_primary:
            return _var_code(G, node_id)
        if G.nodes[node_id]["function_name"].upper() == "ARRAY":
            return _code_array_node(
                G, node_id, conversion_func_sigs, conversion_tracker
            )
        else:
            partial_code_node = partial(
                _code_node,
                G=G,
                is_primary=False,
                conversion_func_sigs=conversion_func_sigs,
                conversion_tracker=conversion_tracker,
            )
            return cc.code_std_function_node(
                G,
                node_id,
                conversion_tracker,
                conversion_func_sigs,
                partial_code_node,
                lambda: None,  # for now no python singatures use templates. implement this following pattern in transpile_sql if want to use that.
                python_special_process_after_code_node,
            )
    if node_type == "table_array":
        _add_table_to_used_tables(G.nodes[node_id]["table_name"])
        return f"df_{_python_safe_name(G.nodes[node_id]['table_name'])}.{_python_safe_name(G.nodes[node_id]['table_column'])}"
    else:
        raise ValueError(f"Unsupported node type: {node_type}")


def python_special_process_after_code_node(
    G, node_id, function_signature, conversion_tracker
):
    if "module" in function_signature:
        _add_module_to_used_modules(function_signature["module"])


def _code_array_node(G, node_id, conversion_func_sigs, conversion_tracker) -> str:
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    ct.update_conversion_tracker_event(conversion_tracker, "code_array_node")
    parents = dags.get_ordered_parent_ids(G, node_id)
    height = int(G.nodes[parents[0]]["value"])
    width = int(G.nodes[parents[1]]["value"])

    # Initialize a list to store rows
    rows = []

    currentNodeIndex = 2  # Starting index for the array values

    for r in range(height):
        this_row_parents = parents[currentNodeIndex : currentNodeIndex + width]
        row_elements = [
            _code_node(G, parent, False, conversion_func_sigs, conversion_tracker)
            for parent in this_row_parents
        ]

        # Build the row string
        row_string = ", ".join(row_elements)
        if width > 1:
            row_string = f"[{row_string}]"

        # Add the row string to the rows list
        rows.append(row_string)

        currentNodeIndex += width

    # Join all the rows into the final code string
    code = "[\n" + ",\n".join(rows) + "\n]"
    return code


##################################
# Section 4: Core transpilation functions
##################################


def convert_and_test(G, conversion_func_sigs, use_tables, conversion_tracker) -> str:
    """
    Core logic.
    Transforms a prepared graph into python.
    Assumes graph has been appropriately prepared
    and results will be tested outside of this function.
    """
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    output_node_ids = G.graph["output_node_ids"]  # already sorted. see is_valid_graph
    all_outputs_as_variables = True if len(output_node_ids) > 1 else False

    dags.mark_nodes_for_caching(
        G,
        usage_count_threshold=3,
        complexity_threshold=200,
        branching_threshold=10,
        all_array_nodes=True,
        all_outputs=all_outputs_as_variables,
        conversion_func_sigs=conversion_func_sigs,
    )

    sorted_nodes = list(nx.topological_sort(G))
    code = ""
    for node_id in sorted_nodes:
        if G.nodes[node_id]["cache"]:
            code += f"{_var_code(G, node_id)} = {_code_node(G, node_id, True, conversion_func_sigs, conversion_tracker)}\n"

    if len(output_node_ids) > 1:
        code += (
            "#returns tuple: ("
            + ", ".join(
                [G.nodes[node_id]["output_name"] for node_id in output_node_ids]
            )
            + ")\n"
        )
        code += (
            "return ("
            + ", ".join([_var_code(G, node_id) for node_id in output_node_ids])
            + ")\n"
        )
    else:
        output_id = output_node_ids[0]
        code += f"return {_code_node(G, output_id, True, conversion_func_sigs, conversion_tracker)}"

    code = _make_header(G, use_tables) + _add_indents(code, 1) + "\n" + "\n"

    return code


def transpile_dags_to_py(
    base_dag_G: nx.MultiDiGraph,
    base_dag_tree,
    function_logic_dags: Dict[str, nx.MultiDiGraph],
    transforms_from_to: Dict[str, nx.MultiDiGraph],
    transforms_protect: Dict[str, nx.MultiDiGraph],
    conversion_func_sigs: Dict[str, Dict],
    library_sigs: Dict[str, List[Dict]],
    auto_add_signatures: bool,
    use_tables: bool,
    conversion_tracker: Dict[str, Any],
) -> str:
    """
    Transpiles XML tree to Python code.

    Args:
        base_dag_G (nx.MultiDiGraph): The graph of the base DAG to be transformed.
        base_dag_tree: The XML tree of the base DAG to be transformed. Used for the test cases embedded in the XML.
        function_logic_dags (Dict[str, nx.MultiDiGraph]): A dictionary mapping function names to their corresponding logic DAGs.
        transforms_from_to (Dict[str, nx.MultiDiGraph]): A dictionary mapping transform function names to their corresponding transform logic DAGs.
        transforms_protect (Dict[str, nx.MultiDiGraph]): A dictionary mapping transform function names to their corresponding protect logic DAGs.
        conversion_func_sigs (Dict[str, Dict]): These are the signatures "allowed" for this conversion, but what that means depends on other settings. Typically in fact this is passed in with an empty dictionary and then the function logic signatures are added to it. However this makes it possible to use the updated version of this as a type of definition of what is done--sort of analagous to a schema file that then can be used going forward to say this is the type of conversions allowed.
        library_func_sigs (Dict[str, List[Dict]]): Library of signatures.
        auto_add_signatures (bool): True to automatically add the signatures from the library to the conversion_func_sigs. False allows user to interactively add. To never add, pass an empty dict in library_func_sigs and then it doesn't matter if this is true or false
        use_tables (bool): True to use tables.
        conversion_tracker (Dict[str, Any]): Tracks conversion including signature mapping dictionary for this conversion.        conversion_tracker (Dict[str, Any]): The conversion tracker for the current conversion.
    Returns:
        str: The transpiled Python code.
    """
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    dags.convert_graph(
        dag_to_convert=base_dag_G,
        function_logic_dags=function_logic_dags,
        transforms_from_to=transforms_from_to,
        transforms_protect=transforms_protect,
        conversion_signatures=conversion_func_sigs,
        library_func_sigs=library_sigs,
        auto_add_signatures=auto_add_signatures,
        conversion_tracker=conversion_tracker,
        renum_nodes=False,
    )

    sigs.if_missing_save_sigs_and_err(conversion_func_sigs, base_dag_G)

    code = convert_and_test(
        base_dag_G, conversion_func_sigs, use_tables, conversion_tracker
    )

    test_results_df = test_code(code, base_dag_tree, base_dag_G)

    true_count = test_results_df["Result"].sum()
    false_count = len(test_results_df) - true_count

    # Evaluating the test results
    if true_count == len(test_results_df):
        print(f"All {true_count} tests passed.")
        return code
    elif false_count == len(test_results_df):
        print(f"All {false_count} tests failed.")
        return ""
    else:
        print(f"{true_count} out of {len(test_results_df)} tests passed.")
        print("The following tests failed:")
        failed_tests = test_results_df.index[~test_results_df["Result"]].tolist()
        print(", ".join(map(str, [i + 1 for i in failed_tests])))

        errs.save_code_results_and_raise_msg(
            code, test_results_df, "Not all tests passed.", "python"
        )
        return ""


def transpile(
    xml_file_name, working_directory, conversion_tracker, override_defaults: Dict
) -> Tuple[str, Dict]:
    """
    Transpiles an XML file to Python code.

    Args:
        xml_file_name (str): The name of the XML file to be transpiled.
        output_file_name (str): The name of the output file where the transpiled code will be saved.
        working_directory (str): The parent directory for function logic and transform subdirectories.

    Returns:
        None
    """
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    data_dict = get_standard_settings(xml_file_name, working_directory)
    data_dict.update(override_defaults)

    if data_dict["use_tables"]:
        dag_to_send: nx.MultiDiGraph = g_tables.pull_out_and_save_tables(
            data_dict, _python_safe_name
        )
    else:
        dag_to_send = data_dict["base_dag_graph"]

    python_code = transpile_dags_to_py(
        base_dag_G=dag_to_send,
        base_dag_tree=data_dict["base_dag_xml_tree"],
        function_logic_dags=data_dict["function_logic_dags"],
        transforms_from_to=data_dict["transforms_from_to"],
        transforms_protect=data_dict["transforms_protect"],
        conversion_func_sigs=data_dict["conversion_func_sigs"],
        library_sigs=data_dict["library_func_sigs"],
        auto_add_signatures=data_dict["auto_add_signatures"],
        use_tables=data_dict["use_tables"],
        conversion_tracker=conversion_tracker,
    )
    return python_code, data_dict["conversion_func_sigs"]


##################################
# Section 5: Test and related functions
##################################


def test_code(code_str, tree, G):
    try:
        exec(code_str, globals())
        function_name = G.graph["name"]
        function_to_test = globals()[function_name]

        graph_input_types = []
        graph_input_names = []
        for node_id in G.graph["input_node_ids"]:
            graph_input_types.append(G.nodes[node_id]["data_type"])
            input_name = _python_safe_name(G.nodes[node_id]["input_name"])
            input_name = f"in_{input_name}"
            graph_input_names.append(input_name)

        graph_output_types = []
        exp_output_names = []
        act_output_names = []
        for node_id in G.graph["output_node_ids"]:
            graph_output_types.append(G.nodes[node_id]["data_type"])
            output_name = _python_safe_name(G.nodes[node_id]["output_name"])
            exp_output_name = f"expected_out_{output_name}"
            act_output_name = f"actual_out_{output_name}"
            exp_output_names.append(exp_output_name)
            act_output_names.append(act_output_name)

        all_column_names = graph_input_names + exp_output_names + act_output_names
        all_column_types = graph_input_types + graph_output_types + graph_output_types

        # Create a mapping from your custom types to pandas/NumPy types
        col_type_mapping = g_tables.convert_col_type_to_pd_types(
            dict(zip(all_column_names, all_column_types))
        )

        num_inputs: int = len(graph_input_names)
        num_outputs = len(graph_output_types)
        num_columns = len(all_column_names) + 1  # add one for results
        num_rows = len(tree.findall("TestCases/test_case"))

        # Initialize the structured array
        dtype = [(name, col_type_mapping[name]) for name in all_column_names]
        dtype.append(("Result", "int"))
        results = np.empty(num_rows, dtype=dtype)

        for test_case_index, test_case in enumerate(
            tree.findall("TestCases/test_case")
        ):
            input_values = []
            for i, input_value in enumerate(test_case.findall("input_value")):
                input_value_converted = _convert_to_type(
                    input_value.attrib["Value"], input_value.attrib["data_type"]
                )
                input_values.append(input_value_converted)
                if not graph_input_types[i] == input_value.attrib["data_type"]:
                    raise ValueError(
                        f"Input type mismatch. Graph input: {graph_input_types[i]}. Test case input: {input_value.attrib['data_type']}"
                    )

            test_result = function_to_test(*input_values)
            row_data: List = [None] * num_columns
            for i, value in enumerate(input_values):
                row_data[i] = value

            expected_outputs = [
                _convert_to_type(
                    output_value.attrib["Value"], output_value.attrib["data_type"]
                )
                for output_value in test_case.findall("output_value")
            ]

            for i, value in enumerate(expected_outputs):
                row_data[num_inputs + i] = value

            if isinstance(test_result, tuple):
                for i, output in enumerate(test_result):
                    row_data[num_inputs + num_outputs + i] = output
                result_match = all(
                    _is_match(r, expected, output_type)
                    for r, expected, output_type in zip(
                        test_result, expected_outputs, graph_output_types
                    )
                )
            else:
                row_data[num_inputs + num_outputs] = test_result
                result_match = _is_match(
                    test_result, expected_outputs[0], graph_output_types[0]
                )
            row_data[num_inputs + num_outputs * 2] = result_match

            for column_index, value in enumerate(row_data):
                results[test_case_index][column_index] = value

    except Exception as e:
        errs.save_code_and_raise_err(code_str, e, "python")
        # Return an empty DataFrame in case of exception
        return pd.DataFrame()

    # Create DataFrame from numpy array
    df = pd.DataFrame(results)
    return df


def _is_match(result_1, result_2, data_type):
    tolerance = 0.01
    if data_type == "Number":
        return abs(result_1 - result_2) < tolerance
    else:
        return result_1 == result_2


#################################
# Section 6: main
#################################


def main() -> None:
    working_directory = "../../../OneDrive/Documents/myDocs/sc_v2_data"
    # xml_file = "ageAtDAte.XML"
    # xml_file = "test_power.XML"
    xml_file = "ranch.XML"
    # xml_file = "CmplxPeriod.XML"
    # xml_file = "test_sum.XML"
    # xml_file = "MultipleMatchInexactNumeric.XML"

    conversion_tracker = ct.empty_conversion_tracker()

    override_defaults = {}

    code, fn_sig_translation_dict = transpile(
        xml_file, working_directory, conversion_tracker, override_defaults
    )

    if code:
        base_file_name = os.path.splitext(xml_file)[0]
        output_file = os.path.join(working_directory, base_file_name + ".py")
        fn_sig_trans_file = os.path.join(
            working_directory, base_file_name + "_func_sigs.json"
        )
        conv_tracker_file = os.path.join(
            working_directory, base_file_name + "_conversion_tracker.json"
        )

        # write code to file
        with open(output_file, "w") as f:
            f.write(code)

        # write function signatures
        with open(fn_sig_trans_file, "w") as f:
            json.dump(fn_sig_translation_dict, f, indent=2)

        # write conversion tracker
        with open(conv_tracker_file, "w") as f:
            json.dump(conversion_tracker, f, indent=2)


if __name__ == "__main__":
    main()
