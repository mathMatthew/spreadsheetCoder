"""
This module takes an sc graph and transpiles it to py spark code
"""

##################################
# Section 1: Imports, constants, global, setup
##################################
import os, time
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    BooleanType,
    DateType, 
    IntegerType,
)
from pyspark.sql import Row

import re
import networkx as nx
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Tuple, List, Literal
from functools import partial
from dotenv import load_dotenv

# internal imports
import dag_tables as g_tables
import setup, validation, errs, dags
import coding_centralized as cc
import conv_tracker as ct
import conversion_rules as cr

language_conversion_rules = "./system_data/spark_supported_functions.json"
# xxx remove global variable and depend on conversion tracker instead.
used_tables = set()
NUMERIC_TOLERANCE = "0.01"
sql_reserved_words_file = "./system_data/spark_reserved_words.txt"

load_dotenv()  # Takes .env variables and adds them to the environment

# make sure to have an .env file with PYSPARK_PYTHON set to your python (likely virtual env for development)
pyspark_python = os.environ.get("PYSPARK_PYTHON", None)
if not pyspark_python:
    raise EnvironmentError("The PYSPARK_PYTHON environment variable is not set.")

with open(sql_reserved_words_file, "r") as file:
    sql_reserved_keywords = {line.strip() for line in file}


def get_standard_settings(base_dag_xml_file, working_directory, mode) -> Dict[str, Any]:

    standard_paths = setup.get_standard_paths(
        base_dag_xml_file, working_directory, "_spark"
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

    # Prepend 'sc_' if the name starts with a digit or is a reserved keyword
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


def _var_code(G, node_id, prefix):
    if "output_name" in G.nodes[node_id]:
        var = _safe_name(G.nodes[node_id]["output_name"])
    elif "node_name" in G.nodes[node_id]:
        var = _safe_name(G.nodes[node_id]["node_name"])
    else:
        var = f"var_{str(node_id)}"

    return _prefix_var_code(var, prefix)

def _prefix_var_code(var_name, prefix):
    if prefix:
        var_name = f"{prefix}.{var_name}"
    return var_name



def _sql_table_name(unformatted_name):
    # wrap table names with this
    # today mainly a placeholder to add extra formatting on table names
    return _safe_name(unformatted_name)


def _sql_column_name(unformatted_name):
    # wrap column names with this
    # today mainly a placeholder to add extra formatting on column names
    return _safe_name(unformatted_name)

from datetime import datetime

def _constant_value_in_code(value, value_type):
    if value_type == "Range":
        raise ValueError("Add support for range type")
    if value_type == "Any":
        raise ValueError("Constant shouldn't have type any")
    if value_type == "Text":
        # Assuming escape_string properly escapes strings for use in PySpark
        return escape_string(value)
    if value_type == "Number":
        try:
            # Directly return the float value
            return float(value)
        except ValueError:
            raise ValueError(f"Invalid number format: {value}")
    if value_type == "Boolean":
        # Convert to Python boolean
        return value.lower() == "true"
    if value_type == "Date":
        try:
            # Convert to Python date object
            date_obj = datetime.strptime(value, "%m/%d/%Y")
            return date_obj.date()
        except ValueError:
            raise ValueError(f"Invalid date format for value '{value}'. Expected format MM/DD/YYYY.")
    
    raise ValueError(f"Unsupported value type: {value_type}")


def convert_to_spark_data_type(data_type):
    if data_type == "Text":
        return StringType()
    elif data_type == "Number":
        return DoubleType()
    elif data_type == "Boolean":
        return BooleanType()
    elif data_type == "Date":
        return DateType()
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


# 2.2 Part 2: These functions take the graph as inputs and then typically use the functions above to generate sql code.
#               The returned sql is used to build the script with the function logic in it.
def _unformatted_primary_table_name(G):
    return "primary_table"
    #return G.graph["name"]

def primary_table_name(G):
    return _sql_table_name(_unformatted_primary_table_name(G))


def _table_name_code(G, node_id):
    return _sql_table_name(G.nodes[node_id]["table_name"])


def _column_name_code(G, node_id):
    if "input_name" in G.nodes[node_id]:
        return _sql_column_name(G.nodes[node_id]["input_name"])
    else:
        return _sql_column_name(G.nodes[node_id]["table_column"])


def _get_output_columns(G):
    output_ids = G.graph["output_node_ids"]
    return [
        (
            _safe_name(G.nodes[node_id]["output_name"]),
            True if G.nodes[node_id]["data_type"] in ["Number", "Date"] else False,
        )
        for node_id in output_ids
    ]


def _initialize_reference_table(spark, table_name, table_definition):
    col_name_type_pairs = list(table_definition["metadata"]["col_types"].items())

    # Define schema
    fields = [
        StructField(_safe_name(col_name), convert_to_spark_data_type(col_type), True)
        for col_name, col_type in col_name_type_pairs
    ]
    schema = StructType(fields)

    data = table_definition["data"]

    num_rows = len(data[col_name_type_pairs[0][0]])

    rows = [
        tuple(
            _constant_value_in_code(data[col_name][i], col_type)
            for col_name, col_type in col_name_type_pairs
        )
        for i in range(num_rows)
    ]

    df = spark.createDataFrame(rows, schema)
    df.createOrReplaceTempView(table_name)
    


def _initialize_reference_tables(spark, G, tree, tables_dict):
    # xxx prior make_header
    assert validation.is_valid_tables_dict(tables_dict), "Invalid tables dict"

    for table_name, table_definition in tables_dict.items():
        _initialize_reference_table(
            spark, table_name, table_definition
        )
        

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
    if placeholder_key == "value":
        return _code_node(G, node_id, True, conversion_rules, conversion_tracker, "")
    if placeholder_key == "primary_table":
        return primary_table_name(G)
    if placeholder_key == "data_type":
        #if we need the spark data type, we'll create a new placeholder_key
        #this one is used of the native data type
        return G.nodes[node_id]["data_type"]
    if placeholder_key.startswith("input"):
        #Handles 3 cases where # can be any number corresponding to the input number for the function, aka parent node
        #"input#_table_name" -> get table name for that input (executes _table_name_code)
        #"input#_col" -> get column name for that input (executes _column_name_code)
        #"input#" -> executes _code_node for that input, no prefix
        #"input#_withprefix_XYZ" -> executes _code_node for that input, with prefix XYZ
        #"input#_withprimarytableprefix" -> executes _code_node for that input using primary_table_name as the prefix
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
                G, parent_node_id, False, conversion_rules, conversion_tracker, ""
            )
        if detail == "table_name":
            return _table_name_code(G, parent_node_id)
        if detail == "col":
            return _column_name_code(G, parent_node_id)
        if detail == "withprimarytableprefix":
            return _code_node(
                G, parent_node_id, False, conversion_rules, conversion_tracker, primary_table_name(G)
            )
        if detail.startswith("withprefix_"):
            prefix = detail.split("_", 2)[1]
            return _code_node(
                G, parent_node_id, False, conversion_rules, conversion_tracker, prefix
            )
    raise KeyError(f"Invalid placeholder key: {placeholder_key}")


def _special_process_after_code_node(
    G, node_id, function_signature, conversion_tracker
):
    # xxxadd support for:_add_functions_to_used_functions(function_signature["add_functions"])
    pass

def _code_node(G, node_id, is_primary, conversion_rules, conversion_tracker, prefix):
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
        var_name = _safe_name(attribs["input_name"])
        return _prefix_var_code(var_name, prefix)
    if node_type == "constant":
        return str(_constant_value_in_code(attribs["value"], data_type))
    if node_type == "function":
        if attribs.get("persist", False) and not is_primary:
            return _var_code(G, node_id, prefix)
        if G.nodes[node_id]["function_name"].upper() == "ARRAY":
            errs.save_dag_and_raise__node(G, node_id, "Add support for ARRAY")
        else:
            partial_code_node = partial(
                _code_node,
                G=G,
                is_primary=False,
                conversion_rules=conversion_rules,
                conversion_tracker=conversion_tracker,
                prefix=prefix,
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
        raise ValueError(
            "Use transforms to avoid needing to code table arrays directly"
        )
    else:
        raise ValueError(f"Unsupported node type: {node_type}")


##################################
# Section 4: Core transpilation functions
##################################


def build_spark_statements(
    G, base_dag_tree, tables_dict, conversion_rules, conversion_tracker
):
    # was: convert_to_sql
    """
    Core logic.
    """

    dags.mark_nodes_to_persist(
        G=G,
        all_outputs=True,
        all_array_nodes=True,
        step_count_trade_off=15,
        conversion_rules=conversion_rules,
        prohibited_types=[],
    )

    spark_statements = []
    sorted_nodes = list(nx.topological_sort(G))
    for node_id in sorted_nodes:
        if G.nodes[node_id].get("persist", False):
            partial_repl_placeholder_fn = partial(
                get_placeholder_val,
                G=G,
                node_id=node_id,
                conversion_rules=conversion_rules,
                conversion_tracker=conversion_tracker,
            )
            spark_statements.append(
                cc.code_persistent_node(
                    G,
                    node_id,
                    conversion_tracker,
                    conversion_rules,
                    partial_repl_placeholder_fn,
                )
            )

    return spark_statements

def run_spark_statements(spark, df, spark_statements, primary_table_name):
    df.printSchema()
    for statement in spark_statements:
        if statement["type"].upper() == "SQL":
            df.createOrReplaceTempView(primary_table_name)
            try:
                # Execute the Spark SQL query
                df = spark.sql(statement["statement"])
            except Exception as e:
                # If an error occurs, print the SQL statement and the error message
                print("An error occurred while executing the following SQL statement:")
                print(statement["statement"])
                print("Error message:", e)
        elif statement["type"].upper() == "WITH-COLUMN-EXPR":
            df = df.withColumn(statement["new_column_name"], expr(statement["expr"]).cast(convert_to_spark_data_type(statement["data_type"])))
        else:
            raise ValueError(f"Unknown statement type: {statement['type']}")
    
    return df

def test_spark_statements(spark, df, spark_statements, primary_table_name, accuracy_summary_query):
    #accuracy summary sql is a sql statement that must return a single row.

    df = run_spark_statements(spark, df, spark_statements, primary_table_name)
    df.createOrReplaceTempView(primary_table_name)

    accuracy_summary_df = spark.sql(accuracy_summary_query)
    
    test_results = accuracy_summary_df.first()
    if test_results["total"] == test_results["all_correct"]:
        print(f"All {test_results['total']} tests passed!")
    else:
        msg = f"{test_results['total'] - test_results['all_correct']} out of {test_results['total']} tests failed."
        df.show()
        raise ValueError(msg)


def _initialize_primary_table_for_test(spark, G, tree):
    # Prepare column metadata
    input_columns = [
        (G.nodes[node_id]["input_name"], G.nodes[node_id]["data_type"])
        for node_id in G.graph["input_node_ids"]
    ]
    predicted_output_columns = [
        (f'{G.nodes[node_id]["output_name"]}_predicted', G.nodes[node_id]["data_type"])
        for node_id in G.graph["output_node_ids"]
    ]

    # Build the schema for table_definition
    col_types = {name: dtype for name, dtype in input_columns + predicted_output_columns}

    # Initialize data dictionary to hold columns of data
    data = {col: [] for col, _ in input_columns + predicted_output_columns}

    # Populate data from test cases
    test_cases = tree.findall("TestCases/test_case")
    if len(test_cases) < 10:
        raise ValueError("Must have at least 10 test cases.")

    for test_case in test_cases:
        for input_value, (name, _) in zip(test_case.findall("input_value"), input_columns):
            data[name].append(input_value.attrib["Value"])

        for output_value, (name, _) in zip(test_case.findall("output_value"), predicted_output_columns):
            data[name].append(output_value.attrib["Value"])

    # Create table_definition
    table_definition = {
        "metadata": {"col_types": col_types},
        "data": data
    }

    # Use the existing function to initialize the DataFrame
    _initialize_reference_table(spark, "primary_table", table_definition)
    return spark.table("primary_table")


def transpile_dags_to_spark_and_test(
    spark,
    base_dag_G: nx.MultiDiGraph,
    base_dag_tree,
    conversion_rules: Dict[str, Any],
    library_sigs: Dict[str, List[Dict]],
    auto_add_signatures: bool,
    conversion_tracker: Dict[str, Any],
):
    """
    Transpiles DAG to spark code.
    validates code
    """

    # initialize tables_dict
    tables_dict = {}

    dags.convert_graph(
        dag_to_convert=base_dag_G,
        conversion_rules=conversion_rules,
        signature_definition_library=library_sigs,
        auto_add_signatures=auto_add_signatures,
        conversion_tracker=conversion_tracker,
        tables_dict=tables_dict,
        separate_tables=True,
        renum_nodes=False,
    )

    cr.if_missing_save_sigs_and_err(conversion_rules, base_dag_G)

    spark_statements = build_spark_statements(
        base_dag_G, base_dag_tree, tables_dict, conversion_rules, conversion_tracker
    )

    conversion_rules = cr.filter_conversion_rules_by_conv_tracker(
        conversion_rules, conversion_tracker
    )

    if len(conversion_rules["functions"]) > 0:
        raise ValueError("Add support for custom functions.")

    # test the generated spark statements on the example test cases
    _initialize_reference_tables(spark, base_dag_G, base_dag_tree, tables_dict)
    df = _initialize_primary_table_for_test(spark, base_dag_G, base_dag_tree)
    
    test_spark_statements(spark, df, spark_statements, primary_table_name(base_dag_G), generate_accuracy_summary_query(base_dag_G))

    return spark_statements, conversion_rules


def test_only(
    spark, xml_file_name, working_directory, mode, conversion_tracker, override_defaults: Dict
):
    data_dict = get_standard_settings(xml_file_name, working_directory, mode)
    data_dict.update(override_defaults)

    spark_statements, conversion_rules = transpile_dags_to_spark_and_test(
        spark,
        base_dag_G=data_dict["base_dag_graph"],
        base_dag_tree=data_dict["base_dag_xml_tree"],
        conversion_rules=data_dict["conversion_rules"],
        library_sigs=data_dict["signature_definition_library"],
        auto_add_signatures=data_dict["auto_add_signatures"],
        conversion_tracker=conversion_tracker,
    )


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


def run_query_single_row_result(test_query, spark) -> Dict[str, Any]:
    try:
        test_results_df = spark.sql(test_query)
    except Exception as e:
        # If there's an error during execution, save the code and re-raise the error
        errs.save_code_and_raise_err(test_query, e, "sql")
        return {}

    result_count = test_results_df.count()

    # Check if exactly one row is returned
    if result_count != 1:
        print(
            f"Error: The test query must be setup to return exactly one row. Instead it returned {result_count} rows."
        )
        return {}

    # Convert the single row to a dictionary and return
    return test_results_df.collect()[0].asDict()


#################################
# Section 6: main
#################################

"""
Overall plan:
1. Convert the DAG to spark statements

2. Prepare reference and test data

3. Test the spark statements

Execute the spark statements on the test data
Compare the results with the expected output in the test cases.

4. Process the full dataset

5. Saving the results
"""

def run_full_data_set(xml_file, complete_conversion_rules_file, override_defaults: Dict) -> None:
    pass

def main() -> None:
    input_directory = "./examples"
    temp_files = "./data/spark_temp_files"
    output_dir = "./data/spark_output_files"

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_output_dir = f"{output_dir}_{timestamp}"

    spark = SparkSession.builder.appName("first app").getOrCreate()
    #this second spark session is not used and i hope to remove it soon.
    #for some unknown reason this prevents getting a long and ugly warning
    #from java about not being able to cleanup temp files. 
    #will monitor and remove this work around when we can.
    spark = SparkSession.builder.appName("second app").getOrCreate()
    
    # main data set.
    data = [
        ("John Doe", 30, 65),
        ("Jane Doe", 25, 60),
        ("Mike Johnson", 60, 65),
        ("Sophia Smith", 64, 65),
    ]
    columns = ["Name", "Age", "Eligible_Retirement_Age"]
    df = spark.createDataFrame(data, schema=columns)

    df = df.withColumn(
        "YearsToRetirement", expr("Greatest(Eligible_Retirement_Age - Age, 0)")
    )

    # Reference dataset
    status_data = [("Gold", 0), ("Silver", 1), ("Bronze", 2)]
    status_columns = ["Status", "YearsToRetirement"]
    status_df = spark.createDataFrame(status_data, schema=status_columns)

    df.createOrReplaceTempView("main")
    status_df.createOrReplaceTempView("status")

    df = spark.sql(
        """
    SELECT 
        m.*, 
        COALESCE(s.Status, 'No Status') as Status
    FROM 
        main m
    LEFT JOIN 
        status s 
    ON 
        m.YearsToRetirement = s.YearsToRetirement
    """
    )

    df.withColumn(
        "New_Status",
        expr("case when YearsToRetirement > 5 then 'plan ahead' else Status end"),
    )

    output_file = os.path.join(unique_output_dir, "final_dataset.csv")
    df.coalesce(1).write.csv(output_file, mode="overwrite", header=True)

    time.sleep(5) #give a small amount of time for processes to shutdown gracefully.
    spark.stop()
    time.sleep(20) #give a small amount of time for processes to shutdown gracefully.
    print("complete")


if __name__ == "__main__":
    main()
