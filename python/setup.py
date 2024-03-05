from typing import Any, Dict, Tuple, List, Optional
import os, json
from lxml import etree
import networkx as nx

# internal imports
import convert_xml as cxml
import validation, dags, errs
import conversion_rules as cr


def update_existing_keys(original_dict, updates_dict):
    """
    Updates the values of existing keys in original_dict based on updates_dict.
    Keys in updates_dict that do not exist in original_dict are ignored.

    Parameters:
    original_dict (dict): The dictionary to be updated.
    updates_dict (dict): The dictionary containing updates.

    Returns:
    dict: The original dictionary with updated values for existing keys.
    """
    return {
        k: updates_dict[k] if k in updates_dict else v for k, v in original_dict.items()
    }


def get_standard_settings(
    paths_dict: Dict[str, str],
    operation_mode: str,
    language_conversion_rules_file: Optional[str],
) -> Dict[str, Any]:
    """
    Initialize and return standard settings based on paths and operation mode.

    Parameters:
        paths_dict (dict): Dictionary of paths required for setup.
        operation_mode (str): Operation mode for the setup. Valid modes are:
            - "complete": Uses the dag_conversion_rules_file from paths_dict exclusively, assuming it contains all
              necessary data. Requires the dag_conversion_rules_file to exist.
            - "supplement": Uses the dag_conversion_rules_file from paths_dictalong with additional data from
              specified directories. Requires the dag_conversion_rules_file to exist.
            - "build": Constructs the dag_conversion_rules_file based on the language_conversion_rules_file along
              with additional data from specified directories. Does not require the dag_conversion_rules_file to
              exist upfront.

    Returns:
        dict: Dictionary containing standard settings.
    """

    # Validate operation mode
    valid_modes = ["complete", "supplement", "build"]
    if operation_mode not in valid_modes:
        raise ValueError(
            f"Invalid operation mode: {operation_mode}. Valid modes are: {', '.join(valid_modes)}."
        )

    if operation_mode in ["build", "supplement"]:
        if language_conversion_rules_file is None:
            raise ValueError(
                "language_conversion_rules_file is required for 'build' and 'supplement' modes."
            )
        if not os.path.exists(language_conversion_rules_file):
            raise FileNotFoundError(
                f"language_conversion_rules_file '{language_conversion_rules_file}' not found."
            )
        lang_conv_rules = cr.load_and_deserialize_rules(language_conversion_rules_file)

    # Check for the existence of the dag_conversion_rules_file in modes that require it
    if operation_mode in ["complete", "supplement"] and not os.path.exists(
        paths_dict["dag_conversion_rules_file"]
    ):
        raise FileNotFoundError(
            f"dag_conversion_rules_file '{paths_dict['dag_conversion_rules_file']}' not found for '{operation_mode}' mode."
        )

    standard_settings: Dict[str, Any] = paths_dict.copy()

    if operation_mode == "complete":
        dag_objects_dict: Dict[str, Any] = initial_dag_objects(
            base_dag_xml_file=paths_dict["xml_file"],
            xsd_file=paths_dict["xsd_file"],
            function_logic_dir="",
            transform_logic_dir="",
            dag_conversion_rules_file=paths_dict["dag_conversion_rules_file"],
            lib_func_sig_dir="",
        )
        standard_settings["auto_add_signatures"] = False

    elif operation_mode == "supplement":

        dag_objects_dict: Dict[str, Any] = initial_dag_objects(
            base_dag_xml_file=paths_dict["xml_file"],
            xsd_file=paths_dict["xsd_file"],
            function_logic_dir=paths_dict["function_logic_dir"],
            transform_logic_dir=paths_dict["transform_logic_dir"],
            dag_conversion_rules_file=paths_dict["dag_conversion_rules_file"],
            lib_func_sig_dir=paths_dict["lib_func_sig_dir"],
        )
        # in supplement mode, dag_conversion_rules takes precendence over language_conversion_rules
        # and language_conversion_rules take precedence over the directory conversion rules.
        # We do this by putting lowest priority first and updating with higher priority.
        conversion_rules = cr.initialize_conversion_rules()
        conversion_rules["function_logic_dags"] = dag_objects_dict[
            "function_logic_dags"
        ]
        conversion_rules["transforms"] = dag_objects_dict["transforms"]
        dag_conversion_rules = dag_objects_dict["conversion_rules"]
        cr.update_conversion_rules(conversion_rules, lang_conv_rules)
        cr.update_conversion_rules(conversion_rules, dag_conversion_rules)

        dag_objects_dict["conversion_rules"] = conversion_rules
        standard_settings["auto_add_signatures"] = True

    elif operation_mode == "build":
        # In 'build' mode, these settings prepare the system to build the dag_conversion_rules_file from available libraries
        dag_objects_dict: Dict[str, Any] = initial_dag_objects(
            base_dag_xml_file=paths_dict["xml_file"],
            xsd_file=paths_dict["xsd_file"],
            function_logic_dir=paths_dict["function_logic_dir"],
            transform_logic_dir=paths_dict["transform_logic_dir"],
            dag_conversion_rules_file="",
            lib_func_sig_dir=paths_dict["lib_func_sig_dir"],
        )
        conversion_rules = cr.initialize_conversion_rules()
        conversion_rules["function_logic_dags"] = dag_objects_dict[
            "function_logic_dags"
        ]
        conversion_rules["transforms"] = dag_objects_dict["transforms"]
        cr.update_conversion_rules(conversion_rules, lang_conv_rules)
        dag_objects_dict["conversion_rules"] = conversion_rules
        standard_settings["auto_add_signatures"] = True

    # To avoid confusion remove these keys since the info if applicable is now in the conversion_rules entry
    # Also, critical to ensure downstream processes have converted.
    del dag_objects_dict["function_logic_dags"]
    standard_settings.update(dag_objects_dict)

    return standard_settings


def add_dir(filename, directory) -> str:
    return os.path.join(directory, filename)


def get_standard_paths(xml_file, working_directory, lang_suffix):
    # lang_suffix examples, '_py', '_sql'
    file_base = os.path.splitext(xml_file)[0]
    conversion_suffix = "_cr"
    dag_conversion_rules_file = f"{file_base}{conversion_suffix}{lang_suffix}.json"
    system_data_dir = "./system_data/"

    paths = {
        "work_dir": working_directory,
        "xml_file": add_dir(xml_file, working_directory),
        "xsd_file": add_dir("sc_5_schema.xsd", system_data_dir),
        "function_logic_dir": add_dir(
            f"function_logic{lang_suffix}", system_data_dir
        ),
        "transform_logic_dir": add_dir(
            f"transform_logic{lang_suffix}", system_data_dir
        ),
        "dag_conversion_rules_file": add_dir(
            dag_conversion_rules_file, working_directory
        ),
        "lib_func_sig_dir": system_data_dir,
    }

    return paths


def initial_dag_objects(
    base_dag_xml_file,
    xsd_file,
    function_logic_dir,
    transform_logic_dir,
    dag_conversion_rules_file,
    lib_func_sig_dir,
) -> Dict[str, Any]:

    with open(xsd_file, "rb") as schema_file:
        schema_root = etree.XML(schema_file.read())  # type: ignore
        schema = etree.XMLSchema(schema_root)

    if os.path.exists(dag_conversion_rules_file):
        conversion_rules = cr.load_and_deserialize_rules(dag_conversion_rules_file)
    else:
        conversion_rules = cr.initialize_conversion_rules()

    if os.path.exists(lib_func_sig_dir):
        # create library of function signatures
        signature_definition_library = cr.initialize_conversion_rules()
        for filename in os.listdir(lib_func_sig_dir):
            if filename.endswith(".json") and filename.startswith("func_sig_"):
                filename = os.path.join(lib_func_sig_dir, filename)
                with open(filename, "r") as file:
                    data = json.load(file)
                cr.add_signatures_to_library(
                    data, signature_definition_library, filename, False
                )
        if not validation.is_valid_signature_definition_dict(
            signature_definition_library, False, True
        ):
            raise ValueError(
                f"Library of function signatures created from {lib_func_sig_dir} is not valid"
            )
    else:
        signature_definition_library = cr.initialize_conversion_rules()

    if os.path.exists(function_logic_dir):
        function_logic_dags: Dict[str, Any] = load_function_logic_dags(
            function_logic_dir, schema
        )
    else:
        function_logic_dags = {}

    transform_schema_file = xsd_file  # for now we just use the same schema file. may want to tighten this in the future with
    # a schema file with additional transform constraints.
    if os.path.exists(transform_logic_dir):
        transforms = load_transform_logic_dags(
            transform_logic_dir, transform_schema_file
        )
    else:
        transforms = {}

    base_dag_xml_tree = cxml.load_3_or_5_tree(base_dag_xml_file)

    if not schema.validate(base_dag_xml_tree.getroot()):
        cxml.save_xml_and_raise(
            base_dag_xml_tree, "XML for base dag file is not valid."
        )

    base_dag_graph: nx.MultiDiGraph = dags.build_nx_graph(base_dag_xml_tree)

    if not validation.is_valid_base_graph(base_dag_graph):
        errs.save_dag_and_raise_message(
            base_dag_graph,
            "XML for base dag file is not valid. Graph validation failed.",
        )

    dict = {
        "base_dag_xml_tree": base_dag_xml_tree,
        "base_dag_graph": base_dag_graph,
        "function_logic_dags": function_logic_dags,
        "transforms": transforms,
        "conversion_rules": conversion_rules,
        "signature_definition_library": signature_definition_library,
    }

    return dict


def load_function_logic_dags(directory, schema) -> Dict[str, Any]:
    def validate_function_inputs(function_logic_dags):
        for dag_name, dag in function_logic_dags.items():
            for node_id in dag.nodes():
                node_attribs = dag.nodes[node_id]
                if node_attribs.get("node_type") == "function":
                    function_name = node_attribs.get("function_name")
                    if function_name in function_logic_dags:
                        matching_dag = function_logic_dags[function_name]
                        # Expected number of inputs is the number of incoming edges in the corresponding DAG
                        actual_inputs = len(matching_dag.graph["input_node_ids"])
                        expected_inputs = dag.in_degree(node_id)
                        if actual_inputs != expected_inputs:
                            errs.save_2dags_and_raise(
                                dag,
                                matching_dag,
                                f"Function '{function_name}' in DAG '{dag_name}' has {expected_inputs} inputs, actual inputs for this defined function is {actual_inputs}. Dag1 = {dag_name}. Dag2 = {matching_dag.graph['name']}",
                            )

    function_logic_dags: Dict[str, Any] = {}

    def add_to_cycle_detection_graph(
        function_logic_dag: nx.MultiDiGraph, cycle_detection_graph: nx.DiGraph
    ):
        from_function_name = function_logic_dag.graph["name"]
        cycle_detection_graph.add_node(from_function_name)
        for node_id in function_logic_dag.nodes():
            attribs = function_logic_dag.nodes[node_id]
            if "function_name" in attribs:
                to_function_name = attribs["function_name"]
                cycle_detection_graph.add_node(to_function_name)
                cycle_detection_graph.add_edge(from_function_name, to_function_name)
        try:
            cycle = nx.find_cycle(cycle_detection_graph, orientation="original")
            # If a cycle is found, raise an error
            raise ValueError(f"Cycle detected: {cycle}")
        except nx.NetworkXNoCycle:
            # If no cycle is found, do nothing
            pass

    cycle_detection_graph = nx.DiGraph()
    for filename in os.listdir(directory):
        if filename.upper().endswith(".XML"):
            tree = cxml.load_3_or_5_tree(os.path.join(directory, filename))

            # build
            function_logic_dag: nx.MultiDiGraph = dags.build_nx_graph(tree)
            name = tree.getroot().get("name").upper()
            if not validation.is_valid_logic_function(
                function_logic_dag, filename, name
            ):
                raise ValueError(
                    f"{filename} XML is not a valid function logic dag. Graph validation failed"
                )
            dags.renumber_nodes(function_logic_dag)
            add_to_cycle_detection_graph(function_logic_dag, cycle_detection_graph)
            function_logic_dags[name] = function_logic_dag

    # check if any node function_name which is also defined here doesn't have the set of inputs expected
    validate_function_inputs(function_logic_dags)

    # save cycle detection graph
    data = nx.node_link_data(cycle_detection_graph)
    with open("./data/cycle_detection_graph.json", "w") as file:
        json.dump(data, file, indent=2)

    return function_logic_dags


def load_transform_logic_dags(
    directory, transform_schema
) -> Dict[str, nx.MultiDiGraph]:
    transforms_dict = {}

    for filename in os.listdir(directory):
        if filename.upper().endswith(".XML"):
            tree = cxml.load_3_or_5_tree(os.path.join(directory, filename))
            transform_logic_dag: nx.MultiDiGraph = dags.build_nx_graph(tree)
            name = tree.getroot().get("name").upper()

            if not validation.is_valid_transform(transform_logic_dag, filename, name):
                errs.save_dag_and_raise_message(
                    transform_logic_dag,
                    f"file {filename} is not a valid transform logic dag. Graph validation failed.",
                )
            else:
                dags.renumber_nodes(transform_logic_dag)
                transforms_dict[name] = transform_logic_dag

    return transforms_dict
