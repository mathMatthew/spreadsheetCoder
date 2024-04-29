"""
This module takes an sc graph and transpiles it to python code
"""

##################################
# Section 1: Imports, constants, global, setup
##################################
import json, os, keyword, dags
import networkx as nx
import pandas as pd
from typing import Any, Dict, Tuple, List
from functools import partial
import pandas as pd
import numpy as np

# internal imports
import setup, validation, errs
import conv_tracker as ct
import coding_centralized as cc
import dag_tables as g_tables
import conversion_rules as cr

INDENT = " " * 2

language_conversion_rules_files = "./system_data/python_supported_functions.json"

# xxx remove global variables and depend on conversion tracker instead.
used_tables = set()
used_functions = set()
used_imports = set()


def get_standard_settings(base_dag_xml_file, working_directory, mode) -> Dict[str, Any]:

    standard_paths = setup.get_standard_paths(
        base_dag_xml_file, working_directory, "_py"
    )

    standard_settings = setup.get_standard_settings(
        standard_paths, mode, language_conversion_rules_files
    )

    standard_settings["use_tables"] = True
    standard_settings["tables_dir"] = os.path.join(working_directory, "tables")

    return standard_settings


##################################
# Section 2: Utilities and python helper functions
##################################
def _save_tables(tables_dir, tables_dict):
    os.makedirs(tables_dir, exist_ok=True)
    pandas_tables = g_tables.convert_to_pandas_tables_dict(tables_dict)

    for table_name, df in pandas_tables.items():
        safe_table_name = _python_safe_name(table_name)

        # Apply _python_safe_name to each column name
        safe_column_names = {col: _python_safe_name(col) for col in df.columns}
        df.rename(columns=safe_column_names, inplace=True)

        # Construct the file name
        file_name = os.path.join(tables_dir, safe_table_name)

        # Save the DataFrame using Pickle
        df.to_pickle(f"{file_name}.pkl")



def _add_functions_to_used_functions(function_names):
    global used_functions
    used_functions.update(function_names)


def _add_imports_to_used_imports(import_statements):
    global used_imports
    used_imports.update(import_statements)


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


def _make_header(G, use_tables, conversion_rules) -> str:
    # build function statements first so we capture additional imports.
    function_statements = []
    for function in used_functions:
        function_definition = conversion_rules["functions"][function]
        function_statements.append(function_definition["text"])
        if "requires_imports" in function_definition:
            _add_imports_to_used_imports(function_definition["requires_imports"])

    if use_tables:
        _add_imports_to_used_imports(["import pandas as pd"])

    tables_dir = G.graph["tables_dir"]
    table_statements = []
    for import_table in used_tables:
        import_table_file_name = os.path.join(
            tables_dir, f"{_python_safe_name(import_table)}.pkl"  
        )
        table_statements.append(
            f"df_{import_table} = pd.read_pickle(r'{import_table_file_name}')\n" 
        )


    main_function_header = f'def {_python_safe_name(G.graph["name"])}('
    # create input names
    input_ids: list[Any] = G.graph[
        "input_node_ids"
    ]  # already sorted. see is_valid_graph.
    input_names = [
        _python_safe_name(G.nodes[node_id]["input_name"]) for node_id in input_ids
    ]
    # Joining the names with a comma
    input_names_str = ", ".join(input_names)
    main_function_header += f"{input_names_str}):\n"

    all_imports = "\n".join(used_imports) + "\n"
    all_tables = "\n".join(table_statements) + "\n"
    all_helper_functions = "\n".join(function_statements) + "\n"
    return all_imports + all_tables + all_helper_functions + main_function_header


##################################
# Section 3: Node processing functions
##################################
def _code_node(G, node_id, is_primary, conversion_rules, conversion_tracker) -> str:
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
        if attribs.get("persist", False) and not is_primary:
            return _var_code(G, node_id)
        if G.nodes[node_id]["function_name"].upper() == "ARRAY":
            return _code_array_node(G, node_id, conversion_rules, conversion_tracker)
        else:
            partial_code_node = partial(
                _code_node,
                G=G,
                is_primary=False,
                conversion_rules=conversion_rules,
                conversion_tracker=conversion_tracker,
            )
            return cc.code_std_function_node(
                G,
                node_id,
                conversion_tracker,
                conversion_rules,
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
    # consider switching to same thing used by transpile_sql. No need for special_process after
    # code because  it just looks at all the used functions at the end and registers them.
    if "add_functions" in function_signature:
        _add_functions_to_used_functions(function_signature["add_functions"])

    if "requires_imports" in function_signature:
        _add_imports_to_used_imports(function_signature["requires_imports"])


def _code_array_node(G, node_id, conversion_rules, conversion_tracker) -> str:
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
            _code_node(G, parent, False, conversion_rules, conversion_tracker)
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


def convert_and_test(G, conversion_rules, use_tables, conversion_tracker) -> str:
    global used_tables, used_functions, used_imports

    used_tables = set()
    used_functions = set()
    used_imports = set()
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

    dags.mark_nodes_to_persist(
        G=G,
        conversion_rules=conversion_rules,
        all_outputs=False,
        all_array_nodes=False,
        step_count_trade_off=5,
        total_steps_threshold=25,
        prohibited_types=[],
    )

    sorted_nodes = list(nx.topological_sort(G))
    code = ""
    for node_id in sorted_nodes:
        if G.nodes[node_id].get("persist", False):
            code += f"{_var_code(G, node_id)} = {_code_node(G, node_id, True, conversion_rules, conversion_tracker)}\n"

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
            + ", ".join(
                [
                    _code_node(G, node_id, False, conversion_rules, conversion_tracker)
                    for node_id in output_node_ids
                ]
            )
            + ")\n"
        )
    else:
        output_id = output_node_ids[0]
        code += f"return {_code_node(G, output_id, True, conversion_rules, conversion_tracker)}"

    code = (
        _make_header(G, use_tables, conversion_rules)
        + _add_indents(code, 1)
        + "\n"
        + "\n"
    )

    return code

def transpile_dags_to_py_and_test(
    base_dag_G: nx.MultiDiGraph,
    base_dag_tree,
    conversion_rules: Dict[str, Dict],
    library_sigs: Dict[str, List[Dict]],
    auto_add_signatures: bool,
    use_tables: bool,
    tables_dir: str,
    conversion_tracker: Dict[str, Any],
) -> Tuple[str, Dict]:
    """
    Transpiles XML tree to Python code then tests all test cases.
    """
    assert validation.is_valid_conversion_tracker(
        conversion_tracker
    ), "Conversion tracker is not valid."

    assert validation.is_valid_conversion_rules_dict(
        conversion_rules
    ), "Conversion rules is not valid."
    assert validation.is_valid_signature_definition_dict(library_sigs, False, True)

    # initialize tables in case they are needed
    tables_dict = {}

    dags.convert_graph(
        dag_to_convert=base_dag_G,
        conversion_rules=conversion_rules,
        signature_definition_library=library_sigs,
        auto_add_signatures=auto_add_signatures,
        conversion_tracker=conversion_tracker,
        tables_dict=tables_dict,
        separate_tables=use_tables,
        renum_nodes=False,
    )

    if use_tables:
        _save_tables(tables_dir, tables_dict)
        base_dag_G.graph["tables_dir"] = tables_dir

    cr.if_missing_save_sigs_and_err(conversion_rules, base_dag_G)

    code = convert_and_test(
        base_dag_G, conversion_rules, use_tables, conversion_tracker
    )

    conversion_rules = cr.filter_conversion_rules_by_conv_tracker(
        conversion_rules, conversion_tracker
    )

    test_results_df = test_code(code, base_dag_tree, base_dag_G)

    true_count = test_results_df["Result"].sum()
    false_count = len(test_results_df) - true_count

    # Evaluating the test results
    if true_count == len(test_results_df):
        print(f"All {true_count} tests passed.")
        return code, conversion_rules
    elif false_count == len(test_results_df):
        print(f"All {false_count} tests failed.")
        return "", conversion_rules
    else:
        print(f"{true_count} out of {len(test_results_df)} tests passed.")
        print("The following tests failed:")
        failed_tests = test_results_df.index[~test_results_df["Result"]].tolist()
        print(", ".join(map(str, [i + 1 for i in failed_tests])))

        errs.save_code_results_and_raise_msg(
            code, test_results_df, "Not all tests passed.", "python"
        )
        return "", conversion_rules


def transpile(
    xml_file_name, working_directory, mode, conversion_tracker, override_defaults: Dict
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

    data_dict = get_standard_settings(xml_file_name, working_directory, mode)
    data_dict.update(override_defaults)

    python_code, conversion_func_sig = transpile_dags_to_py_and_test(
        base_dag_G=data_dict["base_dag_graph"],
        base_dag_tree=data_dict["base_dag_xml_tree"],
        conversion_rules=data_dict["conversion_rules"],
        library_sigs=data_dict["signature_definition_library"],
        auto_add_signatures=data_dict["auto_add_signatures"],
        use_tables=data_dict["use_tables"],
        tables_dir=data_dict["tables_dir"],
        conversion_tracker=conversion_tracker,
    )
    return python_code, conversion_func_sig


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

        test_cases = tree.findall("TestCases/test_case")
        if len(test_cases) < 10:
            raise ValueError("Must have at least 10 test cases.")

        for test_case_index, test_case in enumerate(test_cases):
            input_values = []
            for i, input_value in enumerate(test_case.findall("input_value")):
                # fix this at some point to mirror what transpile_sql is doing here.
                input_value_converted = cc.convert_to_python_type(
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
                # fix this at some point to mirror what transpile_sql is doing here.
                cc.convert_to_python_type(
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

    conversion_tracker = ct.initialize_conversion_tracker()

    override_defaults = {}
    mode = "build"  #'options:  'build' 'complete' 'supplement'

    code, conversion_rules = transpile(
        xml_file, working_directory, mode, conversion_tracker, override_defaults
    )

    if code:
        base_file_name = os.path.splitext(xml_file)[0]
        output_file = os.path.join(working_directory, base_file_name + ".py")
        conversion_rules_file = os.path.join(
            working_directory, base_file_name + "_conversion_rules.json"
        )
        conv_tracker_file = os.path.join(
            working_directory, base_file_name + "_conversion_tracker.json"
        )

        # write code to file
        with open(output_file, "w") as f:
            f.write(code)

        # write function signatures
        with open(conversion_rules_file, "w") as f:
            json.dump(conversion_rules, f, indent=2)

        # write conversion tracker
        with open(conv_tracker_file, "w") as f:
            json.dump(conversion_tracker, f, indent=2)


if __name__ == "__main__":
    main()
