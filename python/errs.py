import networkx as nx
import json

import dag_tables as g_tables

code_types = {"python": {"file_extension": ".py"}, "sql": {"file_extension": ".sql"}}


def save_conversion_rules_and_raise(conversion_rules, msg):
    with open("errors/conversion_rules_temp.json", "w") as f:
        json.dump(conversion_rules, f, indent=2)
    raise ValueError(
        msg + " Function signatures written to errors/conversion_rules_temp.json."
    )


def save_2graphs_plusand_raise(graph_plus1, graph_plus2, message):
    g_tables.save_graph_plus(graph_plus1, ".errors/error_graph_plus1.json")
    g_tables.save_graph_plus(graph_plus2, ".errors/error_graph_plus2.json")
    raise ValueError(
        message + " See errors/error_graph_plus1.json and errors/error_graph_plus2.json"
    )


def save_2dags_and_raise(dag1, dag2, value_error_text):
    data1 = nx.node_link_data(dag1)
    data2 = nx.node_link_data(dag2)

    with open("errors/error_dag1.json", "w") as file:
        json.dump(data1, file, indent=2)
    with open("errors/error_dag2.json", "w") as file:
        json.dump(data2, file, indent=2)
    value_error_text += " Dag1 = errors/error_dag1.json; Dag2 = errors/error_dag2.json"
    raise ValueError(value_error_text)


def save_dag_and_raise_node(G, node_id, message):
    data = nx.node_link_data(G)
    error_file_location = "./errors/error_dag.json"

    with open(error_file_location, "w") as file:
        json.dump(data, file, indent=2)

    message += f" Problem at node {node_id}. Dag saved to {error_file_location}"
    raise ValueError(message)


def save_dag_and_raise_message(G, message):
    data = nx.node_link_data(G)
    error_file_location = "./errors/error_dag.json"

    with open(error_file_location, "w") as file:
        json.dump(data, file, indent=2)

    message += f" Problem with dag. Dag saved to {error_file_location}"
    raise ValueError(message)


def save_code_results_and_raise_msg(code_str, df_results, error_msg, code_type):
    save_code_location = "./errors/error_code" + code_types[code_type]["file_extension"]
    save_results_location = "./errors/results.csv"

    # Save the code string to the file
    with open(save_code_location, "w") as file:
        file.write(code_str)

    # save df_results to csv
    df_results.to_csv(save_results_location, index=False)

    raise ValueError(
        f"{error_msg}. {code_type} code saved to {save_code_location} ; results saved to {save_results_location}"
    )


def save_code_and_raise_err(code_str, error, code_type):
    # Define the location where the code will be saved
    save_location = "./errors/error_code" + code_types[code_type]["file_extension"]

    # Save the code string to the file
    with open(save_location, "w") as file:
        file.write(code_str)
    print(f"{code_type} code saved to {save_location}")
    # Re-raise the original error
    raise error from None


def save_code_and_results_and_raise_msg(code_str, df_results, error_msg, code_type):
    save_location = "./errors/error_code" + code_types[code_type]["file_extension"]
    with open(save_location, "w") as file:
        file.write(code_str)
    df_results.to_csv("errors/error_results.csv", index=False)
    raise ValueError(
        f"{error_msg}. {code_type} code saved to {save_location} and results saved to errors/error_results.csv"
    )
