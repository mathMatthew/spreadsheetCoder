import pandas as pd

def print_dag_details(G):
    """
    Prints the details of the DAG in a tabular format including precedents of each node.

    :param G: The networkx graph object representing the DAG.
    """
    # Prepare data for DataFrame
    data = {
        "Node ID": [],
        "Node Type": [],
        "Function Name": [],
        "Input Order": [],
        "Precedents": [],
    }

    for node in G.nodes(data=True):
        data["Node ID"].append(node[0])
        data["Node Type"].append(node[1].get("node_type", "N/A"))
        data["Function Name"].append(node[1].get("function_name", "N/A"))
        data["Input Order"].append(node[1].get("input_order", "N/A"))
        # Get precedents (predecessors in the graph)
        precedents = list(G.predecessors(node[0]))
        data["Precedents"].append(precedents if precedents else "None")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Print the DataFrame
    print(df.to_string(index=False))
