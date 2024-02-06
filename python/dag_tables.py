"""
this module takes an sc graph and separates out the tables within the graph.
it creates a dictionary that has place for multiple calculation graphs and 
multiple tables that is stored as a json. 
"""

import json, os
import networkx as nx
from typing import Any, Dict, Tuple, List
import pandas as pd

# local imports
import dags
import networkx as nx


class GPlus:
    def __init__(self, graph, tables=None):
        self._graph = graph
        self._tables = tables

    def __getattr__(self, name):
        try:
            return getattr(self._graph, name)
        except AttributeError:
            raise AttributeError(f"'GPlus' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ["_graph", "_tables"]:
            object.__setattr__(self, name, value)
        else:
            setattr(self._graph, name, value)

    def __len__(self):
        return len(self._graph)

    def __getitem__(self, key):
        return self._graph[key]

    def __iter__(self):
        return iter(self._graph)

    def __contains__(self, key):
        return key in self._graph

    def __str__(self):
        return str(self._graph)

    def __repr__(self):
        return repr(self._graph)

    @property
    def tables(self):
        return self._tables

    @tables.setter
    def tables(self, value):
        self._tables = value


def all_precedents_are_constants(G, node):
    for predecessor in G.predecessors(node):
        if G.nodes[predecessor]["node_type"] != "constant":
            return False
    return True


def get_col_and_table_name(input_str):
    # Split the string at the '[' character
    parts = input_str.split("[")
    if len(parts) != 2:
        # Return an error or handle it as you see fit
        raise ValueError("Invalid input string")

    # Extract the table name
    table_name = parts[0]

    # Extract the column name and remove the trailing ']'
    column_name = parts[1].rstrip("]")

    return table_name, column_name


def separate_named_tables(G: nx.MultiDiGraph) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
    G = (
        G.copy()
    )  # type: ignore # given we are passing back the modified graph, ensure no side effects on the input for clarity.
    tables: Dict[str, Dict[str, Any]] = {}
    modification_queue = []
    for node_id in G.nodes:
        node_data = G.nodes[node_id]
        if node_data["node_type"] != "function":
            continue
        if node_data["function_name"].upper() != "ARRAY":
            continue
        if not all_precedents_are_constants(G, node_id):
            continue
        attribs = G.nodes[node_id]
        if attribs["node_name_type"] != "alias":
            continue
        table_name, col_name = get_col_and_table_name(attribs["node_name"])
        table_dict: Dict[str, List[Any]]
        col_type: str
        table_dict, col_type = get_table_dict(G, node_id, col_name)
        add_column(tables, table_name, col_name, col_type, table_dict)
        modification_queue.append((node_id, table_name, col_name, col_type))

    while modification_queue:
        node_id, table_name, col_name, data_type = modification_queue.pop()
        make_table_array_node(G, node_id, table_name, col_name, data_type)

    return G, tables


def add_column(
    tables: Dict[str, Dict[str, Any]], table_name, col_name, col_type, table_dict
):
    # Check if table_name is None
    if table_name is None:
        raise ValueError("table_name is None")

    # Initialize the table and metadata if they don't exist
    if table_name not in tables:
        tables[table_name] = {"data": {}, "metadata": {"col_types": {}}}

    # Check if the column already exists
    if col_name in tables[table_name]["data"]:
        raise ValueError(f"Column '{col_name}' already exists in table '{table_name}'")

    # Add the new column and update metadata
    tables[table_name]["data"][col_name] = table_dict[col_name]
    tables[table_name]["metadata"]["col_types"][col_name] = col_type


def make_table_array_node(G, node_id, table_name, table_column, data_type):
    predecessors = set(G.predecessors(node_id))

    # Remove all edges from each predecessor to the node
    # handling the more than one edge to the same predecessor in this case, is a bit overkill.
    # it can happen in the graph but not in an array node.
    for predecessor in predecessors:
        # Remove all edges from this predecessor to node_id
        while G.has_edge(predecessor, node_id):
            G.remove_edge(predecessor, node_id)

        # Check if the predecessor has no other outgoing edges
        if len(G.out_edges(predecessor)) == 0:
            G.remove_node(predecessor)

    # Set attributes for the node
    if "function_name" in G.nodes[node_id]:
        del G.nodes[node_id]["function_name"]
    G.nodes[node_id]["table_name"] = table_name
    G.nodes[node_id]["table_column"] = table_column
    G.nodes[node_id]["node_type"] = "table_array"
    G.nodes[node_id]["data_type"] = f"TABLE_COLUMN[{data_type}]"


def get_table_dict(
    G: nx.MultiDiGraph, node_id: int, col_name: str
) -> Tuple[Dict[str, List[Any]], str]:
    parents = dags.get_ordered_parent_ids(G, node_id)

    # Check if there are enough nodes to determine array dimensions
    if len(parents) < 3:
        raise ValueError("Not enough predecessors to determine array dimensions")

    # The first two nodes of an array node are constants for height and width
    height = int(G.nodes[parents[0]]["value"])
    width = int(G.nodes[parents[1]]["value"])

    # width must equal 1 since this is a single column
    if width != 1:
        raise ValueError("Array width must be 1")

    # Determine the column data type from the first value node
    first_value_node = parents[2]
    column_data_type: str = G.nodes[first_value_node]["data_type"]

    # Function to parse the value based on the data type
    def parse_value(node):
        data_type = G.nodes[node]["data_type"]
        value = G.nodes[node]["value"]
        if data_type != column_data_type:
            raise ValueError("All items in the array must be of the same data type")

        if data_type == "Number":
            return float(value)
        elif data_type == "Boolean":
            return value.lower() == "true"
        elif data_type == "Date":
            # Assuming the date is in a format that can be directly serialized into JSON
            return value
        elif data_type == "Text" or data_type == "Any":
            return value
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    # Extracting and parsing the values for each cell in the table
    values: List[Any] = [parse_value(node_id) for node_id in parents[2:]]

    # Check if the number of values matches the dimensions
    if len(values) != height * width:
        raise ValueError("Number of values does not match specified dimensions")

    table_dict: Dict[str, List[Any]] = {col_name: values}

    return table_dict, column_data_type


def combine_graph_tables(graph_plus1: GPlus, graph_plus2: GPlus):
    # Merge tables (prioritizing tables from graph_plus1)
    combined_tables = {**graph_plus2["tables"], **graph_plus1["tables"]}

    new_graph_plus1 = GPlus(graph_plus1.graph, combined_tables)
    new_graph_plus2 = GPlus(graph_plus2.graph, combined_tables)

    return (new_graph_plus1, new_graph_plus2)


def create_graph_plus(G) -> GPlus:
    G, tables = separate_named_tables(G)
    return GPlus(G, tables)


def save_graph_plus(graph_plus, output_file):
    graph_plus_json = {
        "graph": nx.node_link_data(graph_plus.graph),  # or other suitable format
        "tables": graph_plus.tables,
    }
    with open(output_file, "w") as file:
        json.dump(graph_plus_json, file, indent=2)


def load_graph_plus_data(input_file) -> GPlus:
    with open(input_file, "r") as file:
        data = json.load(file)
    graph = nx.node_link_graph(data["graph"])  # or other suitable format
    return GPlus(graph, tables=data["tables"])


def main():
    # json_graph_file = "endDateMonths_output.json"
    working_directory = "../../../OneDrive/Documents/myDocs/sc_v2_data"
    xml_file = "singleCriteriaExact.XML"

    # output_file = os.path.splitext(json_graph_file)[0] + "2.json"
    output_file = os.path.splitext(xml_file)[0] + "2.json"
    output_file = os.path.join(working_directory, output_file)

    conversion_tracker = {}

    # json_graph_file = os.path.join(working_directory, json_graph_file)
    G = dags.xml_to_graph(
        xml_file,
        working_directory=working_directory,
        conversion_tracker=conversion_tracker,
        override_defaults={},
    )

    graph_plus = create_graph_plus(G)

    save_graph_plus(graph_plus, output_file)


def convert_col_type_to_pd_types(col_types: Dict[str, str]) -> Dict[str, str]:
    # used as well for numpy array.
    type_mapping = {
        "Text": "object",
        "Number": "float64",
        "Boolean": "bool",
        "Date": "datetime64[s]",  # should i use [ns] here rather than [s], given this is the standard for pandas?
    }
    return {col: type_mapping[ctype] for col, ctype in col_types.items()}


def convert_to_pandas_table(definition: Dict[str, Any]) -> pd.DataFrame:
    metadata = definition["metadata"]
    data = definition["data"]
    dtype_dict = convert_col_type_to_pd_types(metadata["col_types"])
    df = pd.DataFrame(data)
    for col, dtype in dtype_dict.items():
        df[col] = df[col].astype(dtype)  # type: ignore
    return df


def convert_to_pandas_tables_dict(
    tables_dict: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    return {
        table_name: convert_to_pandas_table(definition)
        for table_name, definition in tables_dict.items()
    }


def pull_out_and_save_tables(data_dict, safe_wrapper) -> nx.MultiDiGraph:
    os.makedirs(data_dict["tables_dir"], exist_ok=True)
    modified_dag, tables_dict = separate_named_tables(data_dict["base_dag_graph"])
    pandas_tables = convert_to_pandas_tables_dict(tables_dict)

    for table_name, df in pandas_tables.items():
        # Process the table name through safe_wrapper
        safe_table_name = safe_wrapper(table_name)

        # Apply safe_wrapper to each column name
        safe_column_names = {col: safe_wrapper(col) for col in df.columns}
        df.rename(columns=safe_column_names, inplace=True)

        # Construct the file name
        file_name = os.path.join(data_dict["tables_dir"], safe_table_name)

        # Save the DataFrame
        df.to_parquet(f"{file_name}.parquet")

    modified_dag.graph["tables_dir"] = data_dict["tables_dir"]
    return modified_dag


if __name__ == "__main__":
    main()
