"""
This module takes an sc graph and transpiles it to sql code
"""

##################################
# Section 1: Imports, constants, global, setup
##################################

import json, os, re, sqlite3
import networkx as nx
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Tuple, List, Literal
from functools import partial


# internal imports
import dag_tables as g_tables
import setup, validation, errs, dags
import coding_centralized as cc
import conv_tracker as ct
import conversion_rules as cr


language_conversion_rules = "./system_data/sql_supported_functions.json"
used_tables = set()
NUMERIC_TOLERANCE = "0.01"
sql_reserved_words_file = "./system_data/sql_reserved_words.txt"


with open(sql_reserved_words_file, "r") as file:
    sql_reserved_keywords = {line.strip() for line in file}


def get_standard_settings(base_dag_xml_file, working_directory, mode) -> Dict[str, Any]:

    standard_paths = setup.get_standard_paths(
        base_dag_xml_file, working_directory, "_sql"
    )

    standard_settings = setup.get_standard_settings(
        standard_paths, mode, language_conversion_rules
    )

    return standard_settings


##################################
# Section 2: Utilities and functions returning sql
# items in this section are ordered from roughly by
# closer to the sql/more detailed to further from it
##################################


# 2.1 Part 1: These first versions are closest to the SQL. They often take unformatted names, lists of columns etc.
def _safe_name(name):
    # Convert to upper case to check against reserved keywords
    upper_name = name.upper()

    # Prepend 'col_' if the name starts with a digit or is a reserved keyword
    if upper_name in sql_reserved_keywords or name[0].isdigit():
        name = "sc_" + name

    # Replace invalid characters with underscores
    name = re.sub(r"\W", "_", name)

    return name


def escape_string(value):
    """
    single quotes to double quotes -- escape string for SQL
    """
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return value


def _add_table_to_used_tables(table_name):
    global used_tables
    used_tables.add(table_name)


def _var_code(G, node_id, include_prefix):
    if "output_name" in G.nodes[node_id]:
        var = _safe_name(G.nodes[node_id]["output_name"])
    else:
        var = f"var_{str(node_id)}"

    if include_prefix:
        var = f"{primary_table_name(G)}.{var}"
    return var


def _sql_table_name(unformatted_name):
    # wrap table names with this
    # today mainly a placeholder to add extra formatting on table names
    return _safe_name(unformatted_name)


def _sql_column_name(unformatted_name):
    # wrap column names with this
    # today mainly a placeholder to add extra formatting on column names
    return _safe_name(unformatted_name)


def _constant_value_in_code(value, value_type):
    if value_type == "Range":
        raise ValueError(f"Add support for range type")
    if value_type == "Any":
        raise ValueError(f"Constant shouldn't have type any")
    if value_type == "Text":
        return escape_string(value)
    if value_type == "Number":
        try:
            num_val = float(value)
        except ValueError:
            raise ValueError(f"Invalid number format: {value}")

        if num_val in [float('inf'), float('-inf'), float('nan')]:
            raise ValueError(f"Unsupported special float value: {num_val}")

        return str(num_val)
    if value_type == "Boolean":
        #SQLite uses 1 and 0 for boolean
        if value.lower() == "true":
            return "1"
        else:
            return "0"
    if value_type == "Date":
        date_obj = datetime.strptime(value, "%m/%d/%Y") #the format may depend on your excel settings. may need to modify this.
        return f"'{date_obj.strftime('%Y-%m-%d')}'"

    raise ValueError(f"Unsupported value type: {value_type}")


def convert_to_sql_data_type(
    data_type,
) -> Literal["TEXT", "FLOAT", "BOOLEAN", "DATE"]:
    if data_type == "Text":
        return "TEXT"
    elif data_type == "Number":
        return "FLOAT"
    elif data_type == "Boolean":
        return "BOOLEAN"
    elif data_type == "Date":
        return "TEXT"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def _create_reference_table_sql(table_name, table_definition):
    metadata = table_definition["metadata"]
    col_defs = convert_col_defs(metadata["col_types"])

    create_table_sql = _create_table(table_name, col_defs)

    sql_statements = [create_table_sql]

    # Retrieve the column names and data
    column_names = [col[0] for col in col_defs]
    column_types = [col[1] for col in col_defs]
    data = table_definition["data"]

    # Ensure all columns have the same number of entries
    if not all(len(data[col]) == len(data[column_names[0]]) for col in column_names):
        raise ValueError(
            "All columns must have the same number of entries"
        )

    for i in range(len(data[column_names[0]])):
        record = [
            _constant_value_in_code(data[col_name][i], metadata['col_types'][col_name]) 
            for col_name in column_names
        ]
        insert_sql = _insert_statement(table_name, column_names, record)
        sql_statements.append(insert_sql)

    # Combine all SQL statements into a single string
    combined_sql_code = "\n".join(sql_statements)

    return combined_sql_code

def _insert_statement(unformatted_table_name, unformatted_col_names, record) -> str:
    table_name = _sql_table_name(unformatted_table_name)
    col_names = [_sql_column_name(col_name) for col_name in unformatted_col_names]
    values_str = ", ".join(record)
    columns_str = ", ".join(col_names)
    code = f"INSERT INTO {table_name} ({columns_str}) VALUES ({values_str});"
    return code

def _create_table(unformatted_table_name, col_defs) -> str:
    table_name = _sql_table_name(unformatted_table_name)
    col_names = [
        f"{_safe_name(col_name)} {col_type} {'NULL' if is_nullable else 'NOT NULL'}"
        for col_name, col_type, is_nullable in col_defs
    ]
    col_names_str: str = ", ".join(col_names)
    code = f"CREATE TABLE {table_name} ({col_names_str});"
    return code


def convert_col_defs(col_types):
    col_defs = [
        (col_name, convert_to_sql_data_type(col_type), True)
        for col_name, col_type in col_types.items()
    ]
    return col_defs


# 2.2 Part 2: These functions take the graph as inputs and then typically use the functions above to generate sql code.
#               The returned sql is used to build the script with the function logic in it.
def _unformatted_primary_table_name(G):
    return G.graph["name"]


def primary_table_name(G):
    return _sql_table_name(_unformatted_primary_table_name(G))


def _table_name_code(G, node_id):
    return _sql_table_name(G.nodes[node_id]["table_name"])


def _column_name_code(G, node_id):
    return _sql_column_name(G.nodes[node_id]["table_column"])


def _create_reference_tables(G, tables_dict):
    statements = []
    for table_name, table_definition in tables_dict.items():
        statements.append(_create_reference_table_sql(table_name, table_definition))
    code = "\n\n".join(statements)
    return code


def _make_header(G, tree, tables_dict) -> str:
    statements = []
    statements.append(_create_primary_table_sql(G))
    statements.append(_insert_into_primary_table_sql(G, tree))
    statements.append(_create_reference_tables(G, tables_dict))
    header = "\n\n".join(statements)
    return header


def _insert_into_primary_table_sql(G, tree) -> str:
    statements = []
    input_names = [
        G.nodes[node_id]["input_name"] for node_id in G.graph["input_node_ids"]
    ]
    output_names = [
        f"{G.nodes[node_id]['output_name']}_predicted"
        for node_id in G.graph["output_node_ids"]
    ]
    col_names = (
        ["scenario_id"] + input_names + output_names
    )  # Include 'scenario_id' in the column names

    test_cases = tree.findall("TestCases/test_case")
    if len(test_cases) < 10:
        raise ValueError("Must have at least 10 test cases.")

    scenario_id = 1  # Initialize scenario_id counter
    for test_case in test_cases:
        # Extract input values
        input_values = []
        for input_value in test_case.findall("input_value"):
            input_value_converted = _constant_value_in_code(
                input_value.attrib["Value"], input_value.attrib["data_type"]
            )
            input_values.append(input_value_converted)

        # Extract expected (predicted) output values
        expected_outputs = []
        for output_value in test_case.findall("output_value"):
            output_value_converted = _constant_value_in_code(
                output_value.attrib["Value"], output_value.attrib["data_type"]
            )
            expected_outputs.append(output_value_converted)

        # Combine scenario_id, input values, and expected output values for the insert statement
        all_values = [str(scenario_id)] + input_values + expected_outputs

        statements.append(
            _insert_statement(_unformatted_primary_table_name(G), col_names, all_values)
        )

        scenario_id += 1  # Increment the scenario_id for the next record

    code = "\n".join(statements)
    return code


def _get_output_columns(G):
    output_ids = G.graph["output_node_ids"]
    return [
        (
            _safe_name(G.nodes[node_id]["output_name"]),
            True if G.nodes[node_id]["data_type"] in ["Number", "Date"] else False,
        )
        for node_id in output_ids
    ]


def _create_primary_table_sql(G) -> str:
    # Column tuples: (name, type, is_nullable)
    scenario_col = [("Scenario_ID", "INT", False)]
    input_ids = G.graph["input_node_ids"]
    input_columns = [
        (
            G.nodes[node_id]["input_name"],
            convert_to_sql_data_type(G.nodes[node_id]["data_type"]),
            False,
        )
        for node_id in input_ids
    ]
    var_columns = []
    for node_id in G.nodes:
        if "output_name" in G.nodes[node_id]:
            continue
        if G.nodes[node_id]["cache"]:
            tpl = (
                _var_code(G, node_id, False),
                convert_to_sql_data_type(G.nodes[node_id]["data_type"]),
                True,
            )
            var_columns.append(tpl)

    output_ids = G.graph["output_node_ids"]
    output_columns = [
        (
            G.nodes[node_id]["output_name"],
            convert_to_sql_data_type(G.nodes[node_id]["data_type"]),
            True,
        )
        for node_id in output_ids
    ]
    predicted_output_columns = [
        (
            f'{G.nodes[node_id]["output_name"]}_predicted',
            convert_to_sql_data_type(G.nodes[node_id]["data_type"]),
            True,
        )
        for node_id in output_ids
    ]
    all_columns = (
        scenario_col
        + input_columns
        + var_columns
        + output_columns
        + predicted_output_columns
    )
    code = _create_table(_unformatted_primary_table_name(G), all_columns)
    return code


# 2.3 similar to 2.2, but now we are generating functions we will use as part of testing/displaying the results.


def generate_act_vs_predicted_query(G):
    table_name = primary_table_name(G)
    columns = _get_output_columns(G)

    where_clauses = []
    for col_name, is_numeric in columns:
        if is_numeric:
            where_clauses.append(
                f"ABS({col_name} - {col_name}_predicted) > {NUMERIC_TOLERANCE}"
            )
        else:
            where_clauses.append(f"{col_name} <> {col_name}_predicted")

    where_clause = " OR ".join(where_clauses)
    sql_query = f"SELECT * FROM {table_name} WHERE {where_clause};"

    return sql_query


def generate_input_ouptut_query(G):
    table_name = primary_table_name(G)
    input_columns = [
        _safe_name(G.nodes[node_id]["input_name"])
        for node_id in G.graph["input_node_ids"]
    ]
    output_columns = [
        _safe_name(G.nodes[node_id]["output_name"])
        for node_id in G.graph["output_node_ids"]
    ]
    predicted_output_columns = [
        f'{_safe_name(G.nodes[node_id]["output_name"])}_predicted'
        for node_id in G.graph["output_node_ids"]
    ]
    column_names = input_columns + output_columns + predicted_output_columns
    col_name_str = ", ".join(column_names)
    sql = f"SELECT {col_name_str} FROM {table_name};"
    return sql


def generate_accuracy_summary_query(G):
    table_name = primary_table_name(G)
    columns = _get_output_columns(G)

    case_clauses = []
    all_correct_conditions = []
    for col_name, is_numeric in columns:
        if is_numeric:
            condition = (
                f"(ABS({col_name} - {col_name}_predicted) <= {NUMERIC_TOLERANCE})"
            )
        else:
            condition = f"({col_name} = {col_name}_predicted)"

        case_clauses.append(
            f"SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) as correct_{col_name}"
        )
        all_correct_conditions.append(condition)

    correct_clauses = ", ".join(case_clauses)
    all_correct_clause = " AND ".join(all_correct_conditions)
    all_correct_sum = (
        f"SUM(CASE WHEN {all_correct_clause} THEN 1 ELSE 0 END) as all_correct"
    )

    sql_query = f"SELECT {correct_clauses}, {all_correct_sum}, COUNT(*) as total FROM {table_name};"

    return sql_query


def primary_table_sql(G):
    sql = f"SELECT * FROM {primary_table_name(G)};"
    return sql


##################################
# Section 3: Node processing functions
##################################


def get_placeholder_val(
    placeholder_key, G, node_id, conversion_rules, conversion_tracker
):
    if placeholder_key == "var":
        return _var_code(G, node_id, False)
    if placeholder_key == "prefixed_var":
        return _var_code(G, node_id, True)
    if placeholder_key == "value":
        return _code_node(G, node_id, True, conversion_rules, conversion_tracker)
    if placeholder_key == "primary_table":
        return primary_table_name(G)
    if placeholder_key.startswith("input"):
        without_input = placeholder_key[5:]
        parts = without_input.split("_", 1)
        if len(parts) == 2:
            input_number, detail = parts
        else:
            input_number = parts[0]
            detail = ""
        if not input_number.isdigit():
            raise KeyError(f"Invalid placeholder key: {placeholder_key}")
        input_number = int(input_number)
        parent_node_id = dags.get_ordered_parent_ids(G, node_id)[input_number - 1]
        if not detail:
            return _code_node(
                G, parent_node_id, False, conversion_rules, conversion_tracker
            )
        if detail == "table_name":
            return _table_name_code(G, parent_node_id)
        if detail == "col":
            return _column_name_code(G, parent_node_id)
    raise KeyError(f"Invalid placeholder key: {placeholder_key}")


def _special_process_after_code_node(
    G, node_id, function_signature, conversion_tracker
):
    # xxxadd support for:_add_functions_to_used_functions(function_signature["add_functions"])
    pass


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
        return f'{primary_table_name(G)}.{_safe_name(attribs["input_name"])}'
    if node_type == "constant":
        return _constant_value_in_code(attribs["value"], data_type)
    if node_type == "function":
        if attribs["cache"] and not is_primary:
            return _var_code(G, node_id, True)
        if G.nodes[node_id]["function_name"].upper() == "ARRAY":
            errs.save_dag_and_raise__node(G, node_id, "Add support for ARRAY")
        else:
            partial_code_node = partial(
                _code_node,
                G=G,
                is_primary=False,
                conversion_rules=conversion_rules,
                conversion_tracker=conversion_tracker,
            )
            partial_repl_placeholder_fn = partial(
                get_placeholder_val,
                G=G,
                node_id=node_id,
                conversion_rules=conversion_rules,
                conversion_tracker=conversion_tracker,
            )
            return cc.code_std_function_node(
                G,
                node_id,
                conversion_tracker,
                conversion_rules,
                partial_code_node,
                partial_repl_placeholder_fn,
                _special_process_after_code_node,
            )
    if node_type == "table_array":
        _add_table_to_used_tables(G.nodes[node_id]["table_name"])
        return f"df_{G.nodes[node_id]['table_name']}.{G.nodes[node_id]['table_column']}"
    else:
        raise ValueError(f"Unsupported node type: {node_type}")


##################################
# Section 4: Core transpilation functions
##################################


def convert_to_sql(
    G, base_dag_tree, tables_dict, conversion_rules, conversion_tracker
) -> str:
    """
    Core logic.
    Transforms a prepared graph into python.
    Assumes graph has been appropriately prepared
    and results will be tested outside of this function.
    """

    dags.mark_nodes_for_caching(
        G,
        usage_count_threshold=3,
        complexity_threshold=200,
        branching_threshold=10,
        all_array_nodes=True,
        all_outputs=True,  # SQL created below doesn't have a separate step for writing the output variables. This is instead achieved by marking them for caching.
        conversion_rules=conversion_rules,
    )

    sorted_nodes = list(nx.topological_sort(G))
    code = ""
    for node_id in sorted_nodes:
        if G.nodes[node_id]["cache"]:
            partial_repl_placeholder_fn = partial(
                get_placeholder_val,
                G=G,
                node_id=node_id,
                conversion_rules=conversion_rules,
                conversion_tracker=conversion_tracker,
            )
            code += cc.code_cached_node(
                G,
                node_id,
                conversion_tracker,
                conversion_rules,
                partial_repl_placeholder_fn,
            )

    code = _make_header(G, base_dag_tree, tables_dict) + "\n\n" + code + "\n\n"

    return code


# def execute_closed_dag(
#     G: nx.MultiDiGraph,
#     tables_dict,
#     conversion_rules: Dict[str, List[Dict]],
#     ) -> str:
#     """
#     a 'closed' dag is one with no inputs. only constants.
#     """


def transpile_dags_to_sql_and_test(
    base_dag_G: nx.MultiDiGraph,
    base_dag_tree,
    tables_dict,
    conversion_rules: Dict[str, Any],
    library_sigs: Dict[str, List[Dict]],
    auto_add_signatures: bool,
    conversion_tracker: Dict[str, Any],
) -> Tuple[str, Dict]:
    """
    Transpiles DAG to SQL code.
    """

    dags.convert_graph(
        dag_to_convert=base_dag_G,
        conversion_rules=conversion_rules,
        signature_definition_library=library_sigs,
        auto_add_signatures=auto_add_signatures,
        conversion_tracker=conversion_tracker,
    )

    cr.if_missing_save_sigs_and_err(conversion_rules, base_dag_G)

    code = convert_to_sql(
        base_dag_G, base_dag_tree, tables_dict, conversion_rules, conversion_tracker
    )

    conversion_rules = cr.filter_conversion_rules_by_conv_tracker(
        conversion_rules, conversion_tracker
    )

    conn = sqlite3.connect(":memory:")

    conv_function: Dict[str, Any] = conversion_rules["functions"]
    for func_name, details in conv_function.items():
        num_params, code_str  = details["num_params"], details["text"]
        exec(code_str, globals())  # Define function in global scope
        conn.create_function(
            func_name, num_params, globals()[func_name]
        )  # Register with SQLite

    test_results = test_script(code, generate_accuracy_summary_query(base_dag_G), conn)

    if test_results["total"] == test_results["all_correct"]:
        print(f"All {test_results['total']} tests passed!")
    else:
        msg = f"{test_results['total'] - test_results['all_correct']} out of {test_results['total']} tests failed."
        if len(test_results) > 3:
            for label, value in test_results.items():
                if label != "total" and label != "all_correct":
                    print(f"{label}: {value}")
        df = pd.read_sql_query(primary_table_sql(base_dag_G), conn)
        conn.close()

        errs.save_code_and_results_and_raise_msg(code, df, msg, "sql")
        return "", conversion_rules

    conn.close()
    code += testing_footer(base_dag_G)

    return code, conversion_rules


def transpile(
    xml_file_name, working_directory, mode, conversion_tracker, override_defaults: Dict
) -> Tuple[str, Dict]:
    """
    Transpiles an XML file to SQL code.
    """
    data_dict = get_standard_settings(xml_file_name, working_directory, mode)
    data_dict.update(override_defaults)

    dag_to_send, tables_dict = g_tables.separate_named_tables(
        data_dict["base_dag_graph"]
    )

    sql_code, conversion_rules = transpile_dags_to_sql_and_test(
        base_dag_G=dag_to_send,
        base_dag_tree=data_dict["base_dag_xml_tree"],
        tables_dict=tables_dict,
        conversion_rules=data_dict["conversion_rules"],
        library_sigs=data_dict["signature_definition_library"],
        auto_add_signatures=data_dict["auto_add_signatures"],
        conversion_tracker=conversion_tracker,
    )
    return sql_code, conversion_rules


#################################
# Section 5: Test and related functions
##################################


def testing_footer(G):
    statements = []
    statements.append("--Show summary of actual vs predicted values")
    statements.append(generate_accuracy_summary_query(G))
    statements.append(
        "--Show all records where any predicted value differs from actual"
    )
    statements.append(generate_act_vs_predicted_query(G))
    statements.append(
        "--query of input, output and predicted output fields for all records"
    )
    statements.append(generate_input_ouptut_query(G))
    code = "\n".join(statements)
    return code


def test_script(sql_script, test_query, conn) -> Dict[str, Any]:
    # test_query parameter must be setup to return a single row only.
    print("Testing generated script...")

    try:
        conn.executescript(sql_script)
    except Exception as e:
        # If there's an error during execution, save the code and re-raise the error
        errs.save_code_and_raise_err(sql_script, e, "sql")
        return {}

    cursor = conn.cursor()
    try:
        cursor.execute(test_query)
    except Exception as e:
        # If there's an error during execution, save the code and re-raise the error
        errs.save_code_and_raise_err(test_query, e, "sql")
        return {}

    results = cursor.fetchall()

    # Check if exactly one row is returned
    if len(results) != 1:
        print(
            f"Error: The test query must be setup to return exactly one row. Instead it returned {len(results)} rows."
        )
        return {}

    # Convert the single row to a dictionary
    row = results[0]
    columns = [description[0] for description in cursor.description]
    row_dict = dict(zip(columns, row))
    cursor.close()

    return row_dict


#################################
# Section 6: main
#################################


def main() -> None:
    working_directory = "../../../OneDrive/Documents/myDocs/sc_v2_data"
    output_dir = "../../../sc_output_files"

    # xml_file = "test_power.XML"
    # xml_file = "CmplxPeriod.XML"
    # xml_file = "test_sum.XML"
    # xml_file = "myPandL.XML"
    xml_file = "ranch.XML"
    mode = "build"  #'options:  'build' 'complete' 'supplement'

    conversion_tracker = ct.initialize_conversion_tracker()
    overrides = {}
    code, conversion_rules = transpile(
        xml_file, working_directory, mode, conversion_tracker, overrides
    )

    if code:
        base_file_name = os.path.splitext(xml_file)[0]
        output_file = os.path.join(working_directory, base_file_name + ".SQL")
        conversion_rules_file = os.path.join(
            output_dir, base_file_name + "_cr_sql.json"
        )
        conv_tracker_file = os.path.join(
            working_directory, base_file_name + "_ct_sql.json"
        )

        # write code to file
        with open(output_file, "w") as f:
            f.write(code)
            print(f"Code written to {output_file}")

        # write function signatures
        cr.serialize_and_save_rules(conversion_rules, conversion_rules_file)
        print(f"Conversion rules file written to {conversion_rules_file}")

        # write conversion tracker
        with open(conv_tracker_file, "w") as f:
            json.dump(conversion_tracker, f, indent=2)
            print(f"Conversion tracker written to {conv_tracker_file}")


if __name__ == "__main__":
    main()
